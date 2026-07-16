"""Deterministic H2A CPU-reference benchmark and independent numerical oracles."""

from __future__ import annotations

import hashlib
import math
import time
from typing import Any

import torch

from tt_rqm_kernels.hamiltonian.su2_lowering import lower_two_level_hamiltonian

REPORT_SCHEMA = "tt-rqm-hamiltonian-lowering-report.v1"
BENCHMARK_FAMILY = "HamiltonianLoweringBench"
CASE_IDS = (
    "zero_vector",
    "axis_x",
    "axis_y",
    "axis_z",
    "tiny_norms",
    "random_finite",
    "varying_dt",
    "large_angles",
    "mixed_zero_nonzero",
)
ATOL = 1e-5
RTOL = 1e-5


def analytical_lowering_oracle(
    hamiltonians: torch.Tensor,
    dt: float | torch.Tensor,
    *,
    hbar: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent Float64 implementation of the H2A coefficient equations."""

    coefficients = hamiltonians.detach().cpu().to(torch.float64)
    step = torch.as_tensor(dt, dtype=torch.float64)
    step = torch.broadcast_to(step, coefficients.shape[:2]) / float(hbar)
    scalar = coefficients[..., 0]
    vector = coefficients[..., 1:]
    magnitude = torch.sqrt(torch.sum(vector * vector, dim=-1))
    angle = magnitude * step
    nonzero = magnitude != 0.0
    scale = torch.zeros_like(magnitude)
    scale[nonzero] = torch.sin(angle[nonzero]) / magnitude[nonzero]
    rotor = torch.cat((torch.cos(angle).unsqueeze(-1), vector * scale.unsqueeze(-1)), dim=-1)
    phase_angle = scalar * step
    phase = torch.stack((torch.cos(phase_angle), -torch.sin(phase_angle)), dim=-1)
    return rotor, phase


def matrix_exp_step_oracle(
    hamiltonians: torch.Tensor,
    dt: float | torch.Tensor,
    *,
    hbar: float = 1.0,
) -> torch.Tensor:
    """Build every U(2) step independently with complex128 ``matrix_exp``."""

    coefficients = hamiltonians.detach().cpu().to(torch.float64)
    step = torch.broadcast_to(torch.as_tensor(dt, dtype=torch.float64), coefficients.shape[:2])
    h0, hx, hy, hz = coefficients.unbind(dim=-1)
    zeros = torch.zeros_like(h0)
    matrix = torch.stack(
        (
            torch.stack((torch.complex(h0 + hz, zeros), torch.complex(hx, -hy)), dim=-1),
            torch.stack((torch.complex(hx, hy), torch.complex(h0 - hz, zeros)), dim=-1),
        ),
        dim=-2,
    )
    return torch.linalg.matrix_exp((-1j * step[..., None, None] / float(hbar)) * matrix)


def rotor_phase_matrix(rotor: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """Reconstruct U(2) matrices from the public rotor/phase lane convention."""

    values = rotor.to(torch.float64)
    phases = phase.to(torch.float64)
    w, x, y, z = values.unbind(dim=-1)
    top = torch.stack((torch.complex(w, -z), torch.complex(-y, -x)), dim=-1)
    bottom = torch.stack((torch.complex(y, -x), torch.complex(w, z)), dim=-1)
    su2 = torch.stack((top, bottom), dim=-2)
    scalar = torch.complex(phases[..., 0], phases[..., 1])
    return scalar[..., None, None] * su2


def reference_cases(seed: int = 0) -> list[dict[str, Any]]:
    """Return the frozen deterministic semantic case set."""

    axis_cases: list[dict[str, Any]] = []
    for offset, axis in enumerate(("x", "y", "z"), 1):
        coefficients = torch.zeros((1, 3, 4), dtype=torch.float32)
        coefficients[..., 0] = torch.tensor((0.0, 0.25, -0.5))
        coefficients[..., offset] = torch.tensor((1.0, -2.0, 3.0))
        axis_cases.append({"id": f"axis_{axis}", "hamiltonians": coefficients, "dt": 0.125})

    generator = torch.Generator().manual_seed(seed)
    random_values = torch.randn((4, 7, 4), generator=generator, dtype=torch.float32)
    varying_values = torch.randn((2, 4, 4), generator=generator, dtype=torch.float32)
    mixed = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.5, 1.0, -2.0, 3.0],
                [-0.5, 0.0, 0.0, 0.0],
                [1.0, -4.0, 2.0, 0.5],
            ]
        ],
        dtype=torch.float32,
    )
    return [
        {
            "id": "zero_vector",
            "hamiltonians": torch.tensor(
                [[[2.0, 0.0, 0.0, 0.0], [-3.0, 0.0, 0.0, 0.0]]], dtype=torch.float32
            ),
            "dt": 0.25,
        },
        *axis_cases,
        {
            "id": "tiny_norms",
            "hamiltonians": torch.tensor(
                [[[0.2, 1e-12, 0.0, 0.0], [-0.3, 1e-8, -1e-8, 1e-8]]],
                dtype=torch.float32,
            ),
            "dt": 0.75,
        },
        {"id": "random_finite", "hamiltonians": random_values, "dt": 0.05},
        {
            "id": "varying_dt",
            "hamiltonians": varying_values,
            "dt": torch.tensor(((0.0, 0.01, 0.1, 1.0), (0.5, -0.25, 2.0, 0.125))),
        },
        {
            "id": "large_angles",
            "hamiltonians": torch.tensor(
                [[[1e4, 2e4, -3e4, 4e4], [-2e4, -4e4, 1e4, 3e4]]], dtype=torch.float32
            ),
            "dt": math.pi / 7.0,
        },
        {"id": "mixed_zero_nonzero", "hamiltonians": mixed, "dt": 0.2},
    ]


def run_reference_benchmark(*, seed: int = 0, iterations: int = 5) -> dict[str, Any]:
    """Execute all H2A semantic cases as CPU-reference, non-performance evidence."""

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    results = [_run_case(case, iterations=iterations) for case in reference_cases(seed)]
    return {
        "schema": REPORT_SCHEMA,
        "benchmark_family": BENCHMARK_FAMILY,
        "backend": "torch",
        "execution_label": "cpu_reference",
        "dtype": "float32",
        "seed": seed,
        "iterations": iterations,
        "case_ids": list(CASE_IDS),
        "results": results,
        "stable_benchmark": False,
        "performance_eligible": False,
        "claim_level": None,
        "note": "CPU reference timing only; not Tenstorrent hardware performance evidence.",
    }


def _run_case(case: dict[str, Any], *, iterations: int) -> dict[str, Any]:
    coefficients = case["hamiltonians"]
    dt = case["dt"]
    reference_rotor, reference_phase = analytical_lowering_oracle(coefficients, dt)
    with torch.no_grad():
        lower_two_level_hamiltonian(coefficients, dt)
        started = time.perf_counter()
        for _ in range(iterations):
            rotor, phase = lower_two_level_hamiltonian(coefficients, dt)
        elapsed = time.perf_counter() - started

    rotor64 = rotor.to(torch.float64)
    phase64 = phase.to(torch.float64)
    rotor_abs, rotor_rel, rotor_fail = _error_summary(rotor64, reference_rotor)
    phase_abs, phase_rel, phase_fail = _error_summary(phase64, reference_phase)
    reconstructed = rotor_phase_matrix(rotor64, phase64)
    matrix_reference = matrix_exp_step_oracle(coefficients, dt)
    matrix_error = float(torch.max(torch.abs(reconstructed - matrix_reference)).item())
    nonfinite = int((~torch.isfinite(rotor)).sum().item() + (~torch.isfinite(phase)).sum().item())
    output_bytes = rotor.contiguous().numpy().tobytes() + phase.contiguous().numpy().tobytes()
    return {
        "case_id": case["id"],
        "B": int(coefficients.shape[0]),
        "K": int(coefficients.shape[1]),
        "dt_shape": list(torch.as_tensor(dt).shape),
        "input_hashes": {
            "hamiltonians_sha256": _tensor_sha256(coefficients),
            "dt_sha256": _tensor_sha256(torch.as_tensor(dt, dtype=torch.float32)),
        },
        "latency_ms": elapsed * 1000.0 / iterations,
        "throughput_coefficients_per_s": coefficients.shape[0]
        * coefficients.shape[1]
        * iterations
        / elapsed,
        "max_rotor_absolute_error": rotor_abs,
        "max_rotor_relative_error": rotor_rel,
        "max_phase_absolute_error": phase_abs,
        "max_phase_relative_error": phase_rel,
        "rotor_norm_drift": float(
            torch.max(torch.abs(torch.linalg.vector_norm(rotor64, dim=-1) - 1.0)).item()
        ),
        "phase_norm_drift": float(
            torch.max(torch.abs(torch.linalg.vector_norm(phase64, dim=-1) - 1.0)).item()
        ),
        "complex_matrix_reconstruction_error": matrix_error,
        "failing_value_count": rotor_fail + phase_fail,
        "nonfinite_value_count": nonfinite,
        "checksum": hashlib.sha256(output_bytes).hexdigest(),
        "stable_benchmark": False,
        "performance_eligible": False,
        "execution_label": "cpu_reference",
    }


def _error_summary(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float, int]:
    absolute = torch.abs(actual - expected)
    relative = absolute / torch.clamp(torch.abs(expected), min=1e-30)
    failing = absolute > (ATOL + RTOL * torch.abs(expected))
    return float(absolute.max().item()), float(relative.max().item()), int(failing.sum().item())


def _tensor_sha256(value: torch.Tensor) -> str:
    contiguous = value.detach().cpu().contiguous()
    identity = f"{contiguous.dtype}:{tuple(contiguous.shape)}:".encode()
    return hashlib.sha256(identity + contiguous.numpy().tobytes()).hexdigest()
