"""Fail-closed validation for the first H2A hardware conformance contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCHEMA = "tt-rqm-hamiltonian-lowering-preregistration.v1"
DEFAULT_PREREGISTRATION = Path("benchmarks/manifests/hamiltonian-lowering-h2a-preregistration.json")
PINNED_TT_METAL = "dd2849b5bc6b7a5d38a9eafbeba31ef8d530f8d4"
EXPECTED_CASES = (
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


class HamiltonianLoweringPreregistrationError(ValueError):
    """Raised when the frozen H2A Claim Level 0 contract changes."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HamiltonianLoweringPreregistrationError(message)


def validate_preregistration(payload: Any) -> dict[str, Any]:
    _require(isinstance(payload, dict), "preregistration must be an object")
    data = payload
    _require(data.get("schema") == SCHEMA, "unsupported H2A preregistration schema")
    _require(data.get("status") == "pre_hardware", "H2A must remain pre_hardware")
    _require(data.get("pinned_tt_metal_commit") == PINNED_TT_METAL, "TT-Metal pin changed")
    _require(data.get("target_claim_level") == 0, "first H2A milestone must be Claim Level 0")
    _require(
        data.get("claims")
        == {
            "current_level": None,
            "performance_eligible": False,
            "stable_benchmark": False,
        },
        "pre-hardware H2A cannot carry a hardware or performance claim",
    )
    _require(
        data.get("operation")
        == {
            "family": "HamiltonianLoweringBench",
            "stage": "H2A",
            "input_shape": "[B, K, 4]",
            "input_lane_order": ["h0", "hx", "hy", "hz"],
            "dt": "float32 scalar or broadcastable [B,K]",
            "rotor_shape": "[B, K, 4]",
            "rotor_lane_order": ["w", "x", "y", "z"],
            "phase_shape": "[B, K, 2]",
            "phase_lane_order": ["real", "imag"],
            "dtype": "float32",
            "normalization": "none",
        },
        "H2A operation contract changed",
    )
    cases = data.get("cases")
    _require(isinstance(cases, list), "cases must be a list")
    _require(tuple(case.get("id") for case in cases) == EXPECTED_CASES, "case order changed")
    _require(
        all(case.get("input_identity") == "deterministic_sha256" for case in cases),
        "every H2A case requires deterministic serialized input hashes",
    )
    _require(
        data.get("tolerances")
        == {
            "rotor_atol": 1e-4,
            "rotor_rtol": 1e-4,
            "phase_atol": 1e-4,
            "phase_rtol": 1e-4,
            "failing_values": 0,
            "nonfinite_values": 0,
        },
        "H2A conformance tolerances changed",
    )
    gates = data.get("gates")
    required_gates = {
        "one_wormhole_device",
        "device_identity_recorded",
        "tt_metal_commit_pinned",
        "candidate_binary_sha256_recorded",
        "source_commit_recorded",
        "compiler_runtime_provenance_recorded",
        "deterministic_input_sha256_recorded",
        "whole_output_validated",
        "zero_failing_values",
        "zero_nonfinite_values",
        "no_discarded_or_replaced_designated_result",
    }
    _require(isinstance(gates, list) and set(gates) == required_gates, "H2A gates changed")
    nonclaims = data.get("nonclaims")
    required_nonclaims = {
        "no_hardware_execution_yet",
        "no_performance_claim",
        "no_stability_claim",
        "no_speedup_claim",
        "no_full_h2_end_to_end_claim",
        "no_h1_status_inheritance",
        "no_measured_bandwidth_claim",
        "no_energy_claim",
        "no_dual_device_claim",
        "no_tenstorrent_endorsement",
    }
    _require(
        isinstance(nonclaims, list) and set(nonclaims) == required_nonclaims,
        "H2A nonclaims changed",
    )
    forbidden = {"hardware_report", "release_manifest", "raw_sessions", "performance_cases"}
    _require(not forbidden.intersection(data), "pre-hardware H2A contains evidence fields")
    return data


def load_preregistration(path: Path = DEFAULT_PREREGISTRATION) -> dict[str, Any]:
    return validate_preregistration(json.loads(path.read_text(encoding="utf-8")))
