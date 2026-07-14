"""Validation for the EntanglementDynamicsBench reference preregistration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SCHEMA = "tt-rqm-entanglement-dynamics-preregistration.v1"
DEFAULT_PREREGISTRATION = Path("benchmarks/manifests/entanglement-dynamics-preregistration.json")
EXPECTED_CASES = (
    "product_00",
    "product_01",
    "product_plus_0",
    "bell_phi_plus",
    "bell_phi_minus",
    "bell_psi_plus",
    "bell_psi_minus",
    "local_u2_invariance",
    "xx_interaction",
    "yy_interaction",
    "zz_interaction",
    "heisenberg_interaction",
    "reversed_noncommuting_steps",
    "seeded_time_dependent_chain",
    "float32_chain_drift",
)
EXPECTED_METRICS = (
    "state_norm_error",
    "complex128_max_abs_error",
    "complex128_rms_error",
    "global_phase_aligned_max_abs_error",
    "density_trace_error",
    "density_hermiticity_error",
    "reduced_state_eigenvalue_max_abs_error",
    "subsystem_purity",
    "concurrence",
    "reduced_von_neumann_entropy",
    "nonfinite_values",
)
EXPECTED_NONCLAIMS = {
    "no_hardware_execution_claim",
    "no_claim_level",
    "no_performance_claim",
    "no_stability_claim",
    "no_acceleration_claim",
    "no_throughput_definition",
    "no_hardware_release_manifest",
    "no_full_device_side_lowering_claim",
    "no_tenstorrent_endorsement",
}


class EntanglementPreregistrationError(ValueError):
    """Raised when the reference-only H3 contract changes incompatibly."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EntanglementPreregistrationError(message)


def validate_entanglement_preregistration(payload: Any) -> dict[str, Any]:
    """Validate the exact CPU-reference scope and its public-claim boundary."""

    _require(isinstance(payload, dict), "preregistration must be a JSON object")
    data = payload
    _require(data.get("schema") == SCHEMA, "unsupported preregistration schema")
    _require(data.get("status") == "reference_foundation", "status must remain reference-only")
    _require(data.get("execution_scope") == "cpu_reference", "execution scope changed")

    claims = data.get("claims")
    _require(
        claims
        == {
            "current_level": None,
            "performance_eligible": False,
            "stable_benchmark": False,
        },
        "reference foundation cannot carry a hardware or performance claim",
    )

    operation = data.get("operation")
    _require(isinstance(operation, dict), "operation must be an object")
    expected_operation = {
        "family": "TwoQubitHamiltonianBench",
        "stage": "EntanglementDynamicsBench",
        "coefficient_shape": "[B, K, 4, 4]",
        "pauli_axis_order": ["I", "X", "Y", "Z"],
        "hamiltonian_definition": "H = sum(C[p,q] * sigma[p] tensor sigma[q])",
        "state_shape": "[B, 4, 2]",
        "state_basis_order": ["|00>", "|01>", "|10>", "|11>"],
        "complex_lane_order": ["real", "imag"],
        "flattened_storage_view": "[B, 8]",
        "lowered_operator_shape": "[B, K, 4, 4, 2]",
        "composition_order": "step[K-1] * ... * step[0]",
        "normalization": "none",
    }
    _require(operation == expected_operation, "two-qubit operator contract changed")

    inputs = data.get("input_validation")
    _require(
        inputs
        == {
            "minimum_batch": 1,
            "minimum_steps": 1,
            "coefficients": "finite_real_floating_point",
            "dt": "finite_scalar_or_[B,K]_broadcastable",
            "hbar": "finite_positive",
            "state": "finite_real_lanes_matching_batch_dtype_device",
        },
        "input validation contract changed",
    )

    generator = data.get("generator")
    _require(
        generator == {"dt": 0.05, "hbar": 1.0, "seed": 0},
        "deterministic generator changed",
    )
    cases = data.get("cases")
    _require(isinstance(cases, list), "cases must be a list")
    _require(tuple(case.get("id") for case in cases) == EXPECTED_CASES, "case set changed")
    _require(
        all(case.get("status") == "cpu_reference" for case in cases),
        "every case must remain CPU-reference-only",
    )

    _require(tuple(data.get("metrics", ())) == EXPECTED_METRICS, "metric set changed")
    _require(
        data.get("tolerances")
        == {
            "float64_oracle_atol": 1e-11,
            "float64_oracle_rtol": 1e-11,
            "float32_candidate_atol": 1e-4,
            "float32_candidate_rtol": 1e-4,
            "nonfinite_values": 0,
        },
        "reference tolerances changed",
    )
    _require(
        data.get("dependencies") == {"rqm_entanglement_runtime": False},
        "rqm-entanglement must not become a runtime dependency",
    )
    nonclaims = data.get("nonclaims")
    _require(isinstance(nonclaims, list), "nonclaims must be a list")
    _require(set(nonclaims) == EXPECTED_NONCLAIMS, "required nonclaims changed")

    forbidden = {
        "hardware_release",
        "performance_cases",
        "raw_sessions",
        "plots",
        "throughput",
        "device",
    }
    _require(not forbidden.intersection(data), "hardware or performance fields are forbidden")
    return data


def load_entanglement_preregistration(
    path: Path = DEFAULT_PREREGISTRATION,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return validate_entanglement_preregistration(payload)
