"""Validation for the preregistered SU2ComposeBench contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SCHEMA = "tt-rqm-su2-compose-preregistration.v1"
DEFAULT_PREREGISTRATION = Path("benchmarks/manifests/su2-compose-preregistration.json")
PINNED_TT_METAL = "dd2849b5bc6b7a5d38a9eafbeba31ef8d530f8d4"
EXPECTED_CASES = (
    (32768, 8, "balanced"),
    (8192, 32, "balanced"),
    (2048, 128, "balanced"),
    (512, 512, "balanced"),
    (1024, 128, "scaling"),
    (4096, 128, "scaling"),
    (16384, 128, "scaling"),
    (65536, 128, "scaling"),
)


class PreregistrationError(ValueError):
    """Raised when the SU2ComposeBench preregistration changes incompatibly."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PreregistrationError(message)


def validate_su2_preregistration(payload: Any) -> dict[str, Any]:
    """Validate exact H1 methodology and public-claim boundaries."""

    _require(isinstance(payload, dict), "preregistration must be a JSON object")
    data = payload
    _require(data.get("schema") == SCHEMA, "unsupported SU2ComposeBench schema")
    _require(data.get("status") == "pre_hardware", "foundation must remain pre_hardware")
    _require(data.get("dtype") == "float32", "H1 dtype must remain float32")
    _require(data.get("pinned_tt_metal_commit") == PINNED_TT_METAL, "TT-Metal pin changed")

    operation = data.get("operation")
    _require(isinstance(operation, dict), "operation must be an object")
    _require(operation.get("family") == "SU2HamiltonianBench", "benchmark family changed")
    _require(operation.get("stage") == "SU2ComposeBench", "benchmark stage changed")
    _require(
        operation.get("composition_order") == "step[K-1] * ... * step[0]",
        "composition order changed",
    )
    _require(operation.get("normalization") == "none", "primary output must not normalize")

    cases = data.get("performance_cases")
    _require(isinstance(cases, list), "performance_cases must be a list")
    actual_cases = tuple((case.get("B"), case.get("K"), case.get("series")) for case in cases)
    _require(actual_cases == EXPECTED_CASES, "performance matrix changed")

    methodology = data.get("methodology")
    _require(isinstance(methodology, dict), "methodology must be an object")
    expected_methodology = {
        "device_count": 1,
        "device_id": 0,
        "fused_dispatches_per_chain": 1,
        "unfused_dispatches_per_chain": "K - 1",
        "minimum_steps_per_sample": 2621440,
        "warmup_pairs": 2,
        "samples": 10,
        "paired_order": "alternate_fused_first_by_sample_index",
        "fused_logical_bytes": "24*B*K + 24*B",
        "unfused_logical_bytes": "72*B*(K - 1)",
    }
    _require(methodology == expected_methodology, "preregistered methodology changed")

    tolerances = data.get("tolerances")
    _require(isinstance(tolerances, dict), "tolerances must be an object")
    _require(tolerances.get("hardware_atol") == 1e-4, "hardware atol changed")
    _require(tolerances.get("hardware_rtol") == 1e-4, "hardware rtol changed")
    _require(tolerances.get("cpu_oracle_atol") == 1e-11, "CPU oracle atol changed")
    _require(tolerances.get("cpu_oracle_rtol") == 1e-11, "CPU oracle rtol changed")
    _require(tolerances.get("nonfinite_values") == 0, "nonfinite gate changed")

    claims = data.get("claims")
    _require(isinstance(claims, dict), "claims must be an object")
    _require(claims.get("current_level") is None, "pre-hardware work cannot claim a level")
    _require(claims.get("stable_benchmark") is False, "foundation cannot be stable")
    _require(
        claims.get("public_post_conformance")
        == "RQM runs quantum Hamiltonian simulations on Tenstorrent.",
        "approved public framing changed",
    )
    nonclaims = data.get("nonclaims")
    _require(isinstance(nonclaims, list), "nonclaims must be a list")
    required_nonclaims = {
        "no_stability_claim",
        "no_acceleration_claim",
        "no_cpu_comparison",
        "no_measured_bandwidth_claim",
        "no_energy_claim",
        "no_dual_device_claim",
        "no_full_device_side_hamiltonian_lowering_claim",
        "no_tenstorrent_endorsement",
    }
    _require(set(nonclaims) == required_nonclaims, "required nonclaims changed")
    return data


def load_su2_preregistration(path: Path = DEFAULT_PREREGISTRATION) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return validate_su2_preregistration(payload)
