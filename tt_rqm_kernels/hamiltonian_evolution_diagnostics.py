"""Deterministic numerical diagnostics for H2B large-angle evolution."""

from __future__ import annotations

from array import array
import hashlib
import math
from pathlib import Path
import sys
from typing import Any

import torch

from tt_rqm_kernels.hamiltonian import (
    compose_hamiltonian_matrices,
    lower_two_level_hamiltonian,
    su2_compose_chain,
    u2_matrix_from_rotor_phase,
)
from tt_rqm_kernels.hamiltonian_evolution_domain import (
    PHASE_ANGLE_LIMIT,
    ROTOR_ANGLE_LIMIT,
)

DIAGNOSTIC_SCHEMA = "tt-rqm-h2b-large-angle-diagnostic.v1"
ATOL = 1e-4
RTOL = 1e-4
MATRIX_THRESHOLD = 2e-4


def _f32(value: float | torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float32)


def _split_product(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Emulate the selected H2A Dekker FP32 TwoProduct operation order."""

    splitter = _f32(4097.0)
    a_split = splitter * a
    a_hi = a_split - (a_split - a)
    a_lo = a - a_hi
    b_split = splitter * b
    b_hi = b_split - (b_split - b)
    b_lo = b - b_hi
    hi = a * b
    lo = a_hi * b_hi - hi
    lo = lo + a_hi * b_lo
    lo = lo + a_lo * b_hi
    lo = lo + a_lo * b_lo
    return hi, lo


def _angle_pair(
    coefficient: torch.Tensor, step_hi: torch.Tensor, step_lo: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    hi, lo = _split_product(coefficient, step_hi)
    return hi, lo + coefficient * step_lo


def _reduce_pair(hi: torch.Tensor, lo: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Emulate Candidate B nearest-multiple split-2pi reduction."""

    inverse_two_pi = _f32(float.fromhex("0x1.45f306p-3"))
    negative_two_pi_hi = _f32(float.fromhex("-0x1.921fb6p+2"))
    negative_two_pi_lo = _f32(float.fromhex("0x1.777a5cp-23"))
    negative_two_pi_head = _f32(float.fromhex("-0x1.922p+2"))
    negative_two_pi_tail = _f32(float.fromhex("0x1.28p-16"))
    rounding_bias = _f32(float.fromhex("0x1.8p23"))
    quotient = hi * inverse_two_pi + rounding_bias
    quotient = quotient - rounding_bias
    period_hi = quotient * negative_two_pi_hi
    period_lo = quotient * negative_two_pi_head - period_hi
    period_lo = period_lo + quotient * negative_two_pi_tail
    reduced_hi = hi + period_hi
    reduced_lo = lo + period_lo
    reduced_lo = reduced_lo + quotient * negative_two_pi_lo
    return quotient, reduced_hi + reduced_lo


def compensated_lowering_equivalent(
    hamiltonians: torch.Tensor,
    dt: float | torch.Tensor,
    *,
    hbar: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    """Model the protected compensated H2A arithmetic using ordered FP32 ops."""

    coefficients = hamiltonians.detach().cpu().to(torch.float32)
    dt_value = torch.broadcast_to(torch.as_tensor(dt, dtype=torch.float32), coefficients.shape[:2])
    inverse_hbar = _f32(1.0 / float(hbar))
    flat = coefficients.reshape(-1, 4)
    flat_dt = dt_value.reshape(-1)
    rotors: list[torch.Tensor] = []
    phases: list[torch.Tensor] = []
    traces: list[dict[str, Any]] = []
    for index, row in enumerate(flat):
        h0, hx, hy, hz = row
        step_hi, step_lo = _split_product(flat_dt[index], inverse_hbar)
        r2_xy = hx * hx + hy * hy
        r2 = r2_xy + hz * hz
        magnitude = torch.sqrt(r2)
        safe_magnitude = torch.where(magnitude == 0, _f32(1.0), magnitude)
        reciprocal = torch.reciprocal(safe_magnitude)
        theta_hi, theta_lo = _angle_pair(magnitude, step_hi, step_lo)
        theta_quotient, theta_reduced = _reduce_pair(theta_hi, theta_lo)
        alpha_hi, alpha_lo = _angle_pair(h0, step_hi, step_lo)
        alpha_quotient, alpha_reduced = _reduce_pair(alpha_hi, alpha_lo)
        sin_theta = torch.sin(theta_reduced)
        cos_theta = torch.cos(theta_reduced)
        scale = sin_theta * reciprocal
        vector = torch.stack((hx * scale, hy * scale, hz * scale))
        vector = torch.where(magnitude == 0, torch.zeros_like(vector), vector)
        rotor_w = torch.where(magnitude == 0, _f32(1.0), cos_theta)
        rotor = torch.cat((rotor_w.reshape(1), vector))
        phase = torch.stack((torch.cos(alpha_reduced), -torch.sin(alpha_reduced)))
        rotors.append(rotor)
        phases.append(phase)

        row64 = row.double()
        exact_r2 = torch.sum(row64[1:] * row64[1:])
        exact_r = torch.sqrt(exact_r2)
        exact_step = float(flat_dt[index]) / float(hbar)
        exact_theta = exact_r * exact_step
        exact_alpha = row64[0] * exact_step
        exact_theta_q = torch.round(exact_theta / (2.0 * math.pi))
        exact_alpha_q = torch.round(exact_alpha / (2.0 * math.pi))
        exact_theta_reduced = exact_theta - exact_theta_q * (2.0 * math.pi)
        exact_alpha_reduced = exact_alpha - exact_alpha_q * (2.0 * math.pi)
        traces.append(
            {
                "index": index,
                "r2": _value(r2),
                "r2_error": _value(r2.double() - exact_r2),
                "r": _value(magnitude),
                "r_error": _value(magnitude.double() - exact_r),
                "dt_over_hbar_hi": _value(step_hi),
                "dt_over_hbar_lo": _value(step_lo),
                "theta_hi": _value(theta_hi),
                "theta_lo": _value(theta_lo),
                "theta_product_error": _value(theta_hi.double() + theta_lo.double() - exact_theta),
                "theta_quotient": _value(theta_quotient),
                "theta_exact_quotient": _value(exact_theta_q),
                "theta_reduced": _value(theta_reduced),
                "theta_reduction_error": _value(theta_reduced.double() - exact_theta_reduced),
                "alpha_hi": _value(alpha_hi),
                "alpha_lo": _value(alpha_lo),
                "alpha_product_error": _value(alpha_hi.double() + alpha_lo.double() - exact_alpha),
                "alpha_quotient": _value(alpha_quotient),
                "alpha_exact_quotient": _value(exact_alpha_q),
                "alpha_reduced": _value(alpha_reduced),
                "alpha_reduction_error": _value(alpha_reduced.double() - exact_alpha_reduced),
                "sin_theta": _value(sin_theta),
                "sin_theta_error": _value(sin_theta.double() - torch.sin(exact_theta_reduced)),
                "cos_theta": _value(cos_theta),
                "cos_theta_error": _value(cos_theta.double() - torch.cos(exact_theta_reduced)),
                "sin_alpha": _value(-phase[1]),
                "sin_alpha_error": _value((-phase[1]).double() - torch.sin(exact_alpha_reduced)),
                "cos_alpha": _value(phase[0]),
                "cos_alpha_error": _value(phase[0].double() - torch.cos(exact_alpha_reduced)),
            }
        )
    rotor_tensor = torch.stack(rotors).reshape(*coefficients.shape[:2], 4)
    phase_tensor = torch.stack(phases).reshape(*coefficients.shape[:2], 2)
    return rotor_tensor, phase_tensor, traces


def large_angle_sweep() -> list[dict[str, Any]]:
    """Construct deterministic independent angle, sign, axis, and order probes."""

    cases: list[dict[str, Any]] = []
    directions = {
        "x": (1.0, 0.0, 0.0),
        "y": (0.0, 1.0, 0.0),
        "z": (0.0, 0.0, 1.0),
        "mixed": (2.0 / math.sqrt(29.0), -3.0 / math.sqrt(29.0), 4.0 / math.sqrt(29.0)),
    }
    targets = []
    for turns in (1, 31, 1024, 4096):
        center = 2.0 * math.pi * turns
        targets.extend(
            (
                (f"turn_{turns}_minus", center - 2.0**-10),
                (f"turn_{turns}_exact", center),
                (f"turn_{turns}_plus", center + 2.0**-10),
                (f"half_{turns}_minus", center + math.pi - 2.0**-10),
                (f"half_{turns}_plus", center + math.pi + 2.0**-10),
            )
        )
    for sign in (-1.0, 1.0):
        for target_id, target in targets:
            for direction_id, direction in directions.items():
                dt = 0.25 if direction_id in {"x", "y"} else 0.75
                magnitude = sign * target / dt
                vector = [magnitude * value for value in direction]
                h0 = sign * (target + math.pi / 2.0) / dt
                cases.append(
                    {
                        "id": f"{target_id}_{'neg' if sign < 0 else 'pos'}_{direction_id}",
                        "hamiltonians": torch.tensor([[h0, *vector]], dtype=torch.float32).reshape(
                            1, 1, 4
                        ),
                        "dt": dt,
                        "kind": "single_step_combined",
                    }
                )
    for sign in (-1.0, 1.0):
        angle = sign * (2.0 * math.pi * 4096 + math.pi / 2.0)
        cases.extend(
            (
                {
                    "id": f"phase_only_{'neg' if sign < 0 else 'pos'}",
                    "hamiltonians": torch.tensor([[[angle, 0.0, 0.0, 0.0]]]),
                    "dt": 1.0,
                    "kind": "phase_only",
                },
                {
                    "id": f"rotor_only_{'neg' if sign < 0 else 'pos'}",
                    "hamiltonians": torch.tensor([[[0.0, angle, 0.0, 0.0]]]),
                    "dt": 1.0,
                    "kind": "rotor_only",
                },
            )
        )
    edge = 2.0 * math.pi * 4096 + math.pi / 2.0
    cases.extend(
        (
            {
                "id": "noncommuting_xy_edge",
                "hamiltonians": torch.tensor([[[edge, edge, 0.0, 0.0], [-edge, 0.0, edge, 0.0]]]),
                "dt": 1.0,
                "kind": "noncommuting",
            },
            {
                "id": "noncommuting_yx_edge",
                "hamiltonians": torch.tensor([[[-edge, 0.0, edge, 0.0], [edge, edge, 0.0, 0.0]]]),
                "dt": 1.0,
                "kind": "noncommuting_reversed",
            },
        )
    )
    return cases


def build_large_angle_diagnostic(repo_root: Path) -> dict[str, Any]:
    """Build the stage-separated large-angle report and deterministic sweep."""

    from tt_rqm_kernels.hamiltonian_evolution_benchmark import reference_cases

    case = next(item for item in reference_cases(0) if item["id"] == "large_angle_short_chain")
    hamiltonians = case["hamiltonians"]
    dt = torch.as_tensor(case["dt"], dtype=torch.float32)
    exact_steps = lower_two_level_hamiltonian(hamiltonians.double(), dt.double())
    cpu_steps = lower_two_level_hamiltonian(hamiltonians, dt)
    compensated_steps = compensated_lowering_equivalent(hamiltonians, dt)
    exact_final = su2_compose_chain(*exact_steps)
    cpu_final = su2_compose_chain(*cpu_steps)
    compensated_final = su2_compose_chain(compensated_steps[0], compensated_steps[1])
    oracle = compose_hamiltonian_matrices(hamiltonians, dt)

    retained_root = (
        repo_root
        / "benchmarks/pilots/hamiltonian-lowering-h2a/h2a-clean-reproduction-20260716"
        / "probes/clean-large_angles"
    )
    hardware_rotors = _read_float32(retained_root / "rotors.bin", (1, 2, 4))
    hardware_phases = _read_float32(retained_root / "phases.bin", (1, 2, 2))
    hardware_final = su2_compose_chain(hardware_rotors, hardware_phases)
    retained_report = retained_root / "development-report.json"

    sweep_results = [_sweep_result(item) for item in large_angle_sweep()]
    order_difference = float(
        torch.max(
            torch.abs(
                compose_hamiltonian_matrices(large_angle_sweep()[-2]["hamiltonians"], 1.0)
                - compose_hamiltonian_matrices(large_angle_sweep()[-1]["hamiltonians"], 1.0)
            )
        ).item()
    )
    return {
        "schema": DIAGNOSTIC_SCHEMA,
        "case_id": "large_angle_short_chain",
        "development_report": True,
        "release_evidence": False,
        "claim_level": None,
        "stable_benchmark": False,
        "performance_eligible": False,
        "tolerances": {
            "atol": ATOL,
            "rtol": RTOL,
            "pilot_final_matrix_threshold": MATRIX_THRESHOLD,
        },
        "stages": {
            "A_float64_analytical_lowering": _step_summary(exact_steps, exact_steps),
            "B_float32_cpu_lowering": _step_summary(cpu_steps, exact_steps),
            "C_compensated_h2a_equivalent_cpu": _step_summary(compensated_steps[:2], exact_steps),
            "D_per_step_matrix_reconstruction": {
                "float32_cpu_error": _step_matrix_error(cpu_steps, hamiltonians, dt),
                "compensated_cpu_error": _step_matrix_error(
                    compensated_steps[:2], hamiltonians, dt
                ),
                "retained_compensated_h2a_hardware_error": _step_matrix_error(
                    (hardware_rotors, hardware_phases), hamiltonians, dt
                ),
            },
            "E_float32_h1_composition_of_exact_steps": _final_summary(
                su2_compose_chain(exact_steps[0].float(), exact_steps[1].float()),
                exact_final,
                oracle,
            ),
            "F_float32_h1_composition_of_compensated_steps": {
                "cpu_equivalent": _final_summary(compensated_final, exact_final, oracle),
                "retained_h2a_hardware_steps": _final_summary(hardware_final, exact_final, oracle),
            },
            "G_final_u2_matrix": {
                "uncompensated_float32_error": _matrix_error(cpu_final, oracle),
                "compensated_cpu_error": _matrix_error(compensated_final, oracle),
                "retained_h2a_hardware_steps_error": _matrix_error(hardware_final, oracle),
            },
        },
        "compensated_arithmetic_trace": compensated_steps[2],
        "retained_h2a_hardware_source": {
            "report_path": retained_report.relative_to(repo_root).as_posix(),
            "report_sha256": hashlib.sha256(retained_report.read_bytes()).hexdigest(),
            "candidate_sha256": "b12063fd8ff73ff7372713eeb3fbdea31c56462c94e314713909a1f07e225979",
        },
        "sweep": {
            "case_count": len(sweep_results),
            "case_order": [item["case_id"] for item in sweep_results],
            "results": sweep_results,
            "max_compensated_rotor_error": max(
                item["max_rotor_absolute_error"] for item in sweep_results
            ),
            "max_compensated_phase_error": max(
                item["max_phase_absolute_error"] for item in sweep_results
            ),
            "max_compensated_matrix_error": max(
                item["complex128_final_matrix_error"] for item in sweep_results
            ),
            "noncommuting_reversal_matrix_difference": order_difference,
        },
        "diagnosis": {
            "primary_origin": "uncompensated Float32 angle-product formation",
            "not_primary": [
                "H1 composition",
                "operation order",
                "layout",
                "synchronization",
                "nonfinite arithmetic",
            ],
            "resolution": "The H2B TT-Metal candidate already uses the protected compensated H2A TwoProduct and split-2pi path. Retained compensated H2A hardware steps composed in H1 order have zero combined-tolerance value failures.",
            "acceptance_path": "B_formally_bounded_operating_domain",
            "bounded_domain_required": True,
            "operating_domain": {
                "rotor_angle_limit": ROTOR_ANGLE_LIMIT,
                "phase_angle_limit": PHASE_ANGLE_LIMIT,
                "rotor_last_observed_all_pass_angle": 1536.238807605409,
                "rotor_first_observed_failure_angle": 1539.3794236964986,
                "phase_last_observed_pass_angle": 13176794.633322284,
                "phase_first_observed_failure_angle": 15707963.267948966,
                "large_angle_short_chain_role": "out_of_domain_stress_diagnostic",
            },
        },
    }


def render_large_angle_diagnostic(report: dict[str, Any]) -> str:
    hardware = report["stages"]["F_float32_h1_composition_of_compensated_steps"][
        "retained_h2a_hardware_steps"
    ]
    lines = [
        "# H2B large-angle development diagnostic",
        "",
        "This is a deterministic development report, not release evidence.",
        "",
        "## Diagnosis",
        "",
        "The retained source-foundation failure originates in uncompensated Float32 angle-product formation before trigonometric reduction. H1 composition, order, layout, synchronization, and nonfinite arithmetic are not the primary source.",
        "",
        "The H2B hardware source already uses the protected compensated H2A Dekker TwoProduct and split-2pi reduction. Composing retained compensated H2A N300 step outputs in exact H1 order produces:",
        "",
        f"- rotor max absolute error: `{hardware['max_rotor_absolute_error']:.12g}`",
        f"- phase max absolute error: `{hardware['max_phase_absolute_error']:.12g}`",
        f"- failing values at atol=rtol=1e-4: `{hardware['failing_value_count']}`",
        f"- complex128 final-matrix error: `{hardware['complex128_final_matrix_error']:.12g}`",
        "",
        "This resolves the original failure mechanism, but the wider mixed-direction sweep finds a separate magnitude-formation boundary. The pilot therefore selects acceptance path B.",
        "",
        f"Frozen pilot domain: `abs(theta) <= {ROTOR_ANGLE_LIMIT}` and `abs(alpha) <= {PHASE_ANGLE_LIMIT}` radians for every logical step. The public mathematical API remains valid outside this implementation conformance domain.",
        "",
        "## Sweep",
        "",
        f"The deterministic sweep contains `{report['sweep']['case_count']}` cases spanning signs, h0, vector magnitude, direction, dt, step count, commuting axes, noncommuting axes, integer/half-integer pi neighborhoods, 2pi multiples, quotient boundaries, and cancellation-sensitive reductions.",
        "",
        f"Maximum compensated CPU-equivalent rotor error: `{report['sweep']['max_compensated_rotor_error']:.12g}`  ",
        f"Maximum compensated CPU-equivalent phase error: `{report['sweep']['max_compensated_phase_error']:.12g}`  ",
        f"Maximum compensated CPU-equivalent matrix error: `{report['sweep']['max_compensated_matrix_error']:.12g}`",
        "",
        "The machine-readable report contains every stage, arithmetic trace, quotient, reduced angle, trigonometric error, and sweep result.",
    ]
    return "\n".join(lines) + "\n"


def _sweep_result(case: dict[str, Any]) -> dict[str, Any]:
    hamiltonians = case["hamiltonians"].to(torch.float32)
    dt = torch.as_tensor(case["dt"], dtype=torch.float32)
    exact_steps = lower_two_level_hamiltonian(hamiltonians.double(), dt.double())
    compensated = compensated_lowering_equivalent(hamiltonians, dt)
    actual = su2_compose_chain(compensated[0], compensated[1])
    expected = su2_compose_chain(*exact_steps)
    oracle = compose_hamiltonian_matrices(hamiltonians, dt)
    summary = _final_summary(actual, expected, oracle)
    return {"case_id": case["id"], "kind": case["kind"], **summary}


def _step_summary(
    actual: tuple[torch.Tensor, torch.Tensor], expected: tuple[torch.Tensor, torch.Tensor]
) -> dict[str, Any]:
    rotor_error = torch.abs(actual[0].double() - expected[0].double())
    phase_error = torch.abs(actual[1].double() - expected[1].double())
    return {
        "max_rotor_absolute_error": float(rotor_error.max()),
        "max_phase_absolute_error": float(phase_error.max()),
        "failing_value_count": _failure_count(actual, expected),
        "nonfinite_value_count": int(
            (~torch.isfinite(actual[0])).sum() + (~torch.isfinite(actual[1])).sum()
        ),
    }


def _final_summary(
    actual: tuple[torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor],
    oracle: torch.Tensor,
) -> dict[str, Any]:
    result = _step_summary(actual, expected)
    result["complex128_final_matrix_error"] = _matrix_error(actual, oracle)
    return result


def _failure_count(
    actual: tuple[torch.Tensor, torch.Tensor], expected: tuple[torch.Tensor, torch.Tensor]
) -> int:
    count = 0
    for left, right in zip(actual, expected, strict=True):
        error = torch.abs(left.double() - right.double())
        count += int((error > (ATOL + RTOL * torch.abs(right.double()))).sum())
    return count


def _matrix_error(actual: tuple[torch.Tensor, torch.Tensor], oracle: torch.Tensor) -> float:
    matrix = u2_matrix_from_rotor_phase(actual[0].double(), actual[1].double())
    return float(torch.max(torch.abs(matrix - oracle)).item())


def _step_matrix_error(
    actual: tuple[torch.Tensor, torch.Tensor],
    hamiltonians: torch.Tensor,
    dt: torch.Tensor,
) -> float:
    matrix = u2_matrix_from_rotor_phase(actual[0].double(), actual[1].double())
    coefficients = hamiltonians.double()
    dt64 = torch.broadcast_to(dt.double(), coefficients.shape[:2])
    h0, hx, hy, hz = coefficients.unbind(-1)
    zeros = torch.zeros_like(h0)
    hmatrix = torch.stack(
        (
            torch.stack((torch.complex(h0 + hz, zeros), torch.complex(hx, -hy)), -1),
            torch.stack((torch.complex(hx, hy), torch.complex(h0 - hz, zeros)), -1),
        ),
        -2,
    )
    reference = torch.linalg.matrix_exp(-1j * dt64[..., None, None] * hmatrix)
    return float(torch.max(torch.abs(matrix - reference)).item())


def _read_float32(path: Path, shape: tuple[int, ...]) -> torch.Tensor:
    values = array("f")
    values.frombytes(path.read_bytes())
    if sys.byteorder != "little":
        values.byteswap()
    return torch.tensor(values, dtype=torch.float32).reshape(shape)


def _value(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())
