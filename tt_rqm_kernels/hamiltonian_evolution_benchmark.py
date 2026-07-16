"""Deterministic H2B reference benchmark and whole-chain diagnostics."""

from __future__ import annotations

import hashlib
import math
import time
from typing import Any

import torch

from tt_rqm_kernels.hamiltonian.su2_evolution import evolve_two_level_hamiltonian
from tt_rqm_kernels.hamiltonian.su2_reference import (
    compose_hamiltonian_matrices,
    u2_matrix_from_rotor_phase,
)

REPORT_SCHEMA = "tt-rqm-hamiltonian-evolution-report.v1"
BENCHMARK_FAMILY = "HamiltonianEvolutionBench"
CASE_IDS = (
    "identity_k1",
    "zero_vector_phase_chain",
    "axis_x",
    "axis_y",
    "axis_z",
    "noncommuting_xy",
    "noncommuting_yx",
    "mixed_zero_nonzero",
    "tiny_norms",
    "varying_dt",
    "random_finite",
    "long_chain",
    "large_angle_short_chain",
)
ATOL = 1e-5
RTOL = 1e-5


def reference_cases(seed: int = 0) -> list[dict[str, Any]]:
    """Return deterministic semantic cases spanning K=1 through K=512."""

    def axis(lane: int, steps: int, scale: float = 1.0) -> torch.Tensor:
        values = torch.zeros((1, steps, 4), dtype=torch.float32)
        values[..., 0] = torch.linspace(-0.3, 0.3, steps)
        values[..., lane] = torch.linspace(0.25, scale, steps)
        return values

    xy = torch.tensor([[[0.1, 0.8, 0.0, 0.0], [-0.2, 0.0, 1.1, 0.0]]])
    yx = xy.flip(1).clone()
    mixed = torch.zeros((2, 32, 4), dtype=torch.float32)
    mixed[:, 1::3, :] = torch.tensor((0.2, 0.5, -0.25, 0.75))
    mixed[:, 2::5, :] = torch.tensor((-0.1, -0.4, 0.6, 0.2))
    tiny = torch.zeros((1, 128, 4), dtype=torch.float32)
    tiny[..., 0] = 0.01
    tiny[..., 1] = torch.linspace(1e-12, 1e-8, 128)
    tiny[..., 2] = -tiny[..., 1]
    varying = axis(3, 8, 1.5).repeat(2, 1, 1)
    varying_dt = torch.linspace(-0.2, 0.3, 16, dtype=torch.float32).reshape(2, 8)
    generator = torch.Generator().manual_seed(seed)
    random_values = 0.5 * torch.randn((3, 32, 4), generator=generator)
    long_values = 0.04 * torch.randn((1, 512, 4), generator=generator)
    return [
        {"id": "identity_k1", "hamiltonians": torch.zeros((2, 1, 4)), "dt": 0.25},
        {
            "id": "zero_vector_phase_chain",
            "hamiltonians": axis(1, 8) * torch.tensor((1.0, 0.0, 0.0, 0.0)),
            "dt": 0.125,
        },
        {"id": "axis_x", "hamiltonians": axis(1, 2), "dt": 0.2},
        {"id": "axis_y", "hamiltonians": axis(2, 8), "dt": 0.1},
        {"id": "axis_z", "hamiltonians": axis(3, 32), "dt": 0.025},
        {"id": "noncommuting_xy", "hamiltonians": xy.float(), "dt": 0.7},
        {"id": "noncommuting_yx", "hamiltonians": yx.float(), "dt": 0.7},
        {"id": "mixed_zero_nonzero", "hamiltonians": mixed, "dt": 0.05},
        {"id": "tiny_norms", "hamiltonians": tiny, "dt": 0.75},
        {"id": "varying_dt", "hamiltonians": varying, "dt": varying_dt},
        {"id": "random_finite", "hamiltonians": random_values, "dt": 0.03},
        {"id": "long_chain", "hamiltonians": long_values, "dt": 0.02},
        {
            "id": "large_angle_short_chain",
            "hamiltonians": torch.tensor(
                [[[1e4, 2e4, -3e4, 4e4], [-2e4, -4e4, 1e4, 3e4]]],
                dtype=torch.float32,
            ),
            "dt": math.pi / 7.0,
        },
    ]


def run_reference_benchmark(*, seed: int = 0, iterations: int = 3) -> dict[str, Any]:
    """Run H2B CPU reference cases as non-performance conformance evidence."""

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    cases = reference_cases(seed)
    results = [_run_case(case, iterations=iterations) for case in cases]
    by_id = {case["id"]: case for case in cases}
    xy = compose_hamiltonian_matrices(by_id["noncommuting_xy"]["hamiltonians"], 0.7)
    yx = compose_hamiltonian_matrices(by_id["noncommuting_yx"]["hamiltonians"], 0.7)
    order_sentinel = float(torch.max(torch.abs(xy - yx)).item())
    if order_sentinel <= 1e-6:
        raise RuntimeError("noncommuting order sentinel did not distinguish reversed input")
    return {
        "schema": REPORT_SCHEMA,
        "benchmark_family": BENCHMARK_FAMILY,
        "backend": "torch",
        "execution_label": "cpu_reference",
        "dtype": "float32",
        "seed": seed,
        "iterations": iterations,
        "case_ids": list(CASE_IDS),
        "chain_lengths": sorted({int(case["hamiltonians"].shape[1]) for case in cases}),
        "noncommuting_reversal_matrix_difference": order_sentinel,
        "results": results,
        "stable_benchmark": False,
        "performance_eligible": False,
        "claim_level": None,
        "note": "CPU reference timing only; no hardware, stability, or performance claim.",
    }


def _run_case(case: dict[str, Any], *, iterations: int) -> dict[str, Any]:
    coefficients = case["hamiltonians"]
    dt = case["dt"]
    reference_dt = torch.as_tensor(dt, dtype=coefficients.dtype).double()
    reference_rotor, reference_phase = evolve_two_level_hamiltonian(
        coefficients.double(), reference_dt
    )
    with torch.no_grad():
        evolve_two_level_hamiltonian(coefficients, dt)
        started = time.perf_counter()
        for _ in range(iterations):
            rotor, phase = evolve_two_level_hamiltonian(coefficients, dt)
        elapsed = time.perf_counter() - started

    rotor64, phase64 = rotor.double(), phase.double()
    rotor_abs, rotor_rel, rotor_fail = _error_summary(rotor64, reference_rotor)
    phase_abs, phase_rel, phase_fail = _error_summary(phase64, reference_phase)
    actual_matrix = u2_matrix_from_rotor_phase(rotor64, phase64)
    oracle_matrix = compose_hamiltonian_matrices(coefficients, dt)
    direct_error = float(torch.max(torch.abs(actual_matrix - oracle_matrix)).item())
    global_error = _global_phase_aware_error(actual_matrix, oracle_matrix)
    nonfinite = int((~torch.isfinite(rotor)).sum() + (~torch.isfinite(phase)).sum())
    output = rotor.contiguous().numpy().tobytes() + phase.contiguous().numpy().tobytes()
    return {
        "case_id": case["id"],
        "B": int(coefficients.shape[0]),
        "K": int(coefficients.shape[1]),
        "dt_shape": list(torch.as_tensor(dt).shape),
        "latency_ms": elapsed * 1000.0 / iterations,
        "max_rotor_absolute_error": rotor_abs,
        "max_rotor_relative_error": rotor_rel,
        "max_phase_absolute_error": phase_abs,
        "max_phase_relative_error": phase_rel,
        "failing_value_count": rotor_fail + phase_fail,
        "nonfinite_value_count": nonfinite,
        "rotor_norm_drift": float(
            torch.max(torch.abs(torch.linalg.vector_norm(rotor64, dim=-1) - 1.0)).item()
        ),
        "phase_norm_drift": float(
            torch.max(torch.abs(torch.linalg.vector_norm(phase64, dim=-1) - 1.0)).item()
        ),
        "complex128_final_matrix_error": direct_error,
        "direct_final_matrix_error": direct_error,
        "global_phase_aware_final_matrix_error": global_error,
        "checksum": hashlib.sha256(output).hexdigest(),
        "stable_benchmark": False,
        "performance_eligible": False,
        "claim_level": None,
    }


def _error_summary(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float, int]:
    absolute = torch.abs(actual - expected)
    relative = absolute / torch.clamp(torch.abs(expected), min=1e-30)
    failing = absolute > (ATOL + RTOL * torch.abs(expected))
    return float(absolute.max()), float(relative.max()), int(failing.sum())


def _global_phase_aware_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    overlap = torch.sum(torch.conj(expected) * actual, dim=(-2, -1))
    unit_phase = torch.where(overlap.abs() > 0, overlap / overlap.abs(), torch.ones_like(overlap))
    aligned = actual * torch.conj(unit_phase)[..., None, None]
    return float(torch.max(torch.abs(aligned - expected)).item())
