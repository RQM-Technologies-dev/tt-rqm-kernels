from __future__ import annotations

import copy

import pytest

from tt_rqm_kernels.entanglement_benchmark import (
    EntanglementPreregistrationError,
    load_entanglement_preregistration,
    validate_entanglement_preregistration,
)


def test_committed_entanglement_preregistration_is_valid() -> None:
    manifest = load_entanglement_preregistration()

    assert manifest["status"] == "reference_foundation"
    assert manifest["claims"] == {
        "current_level": None,
        "performance_eligible": False,
        "stable_benchmark": False,
    }


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("operation", "composition_order", "step[0] * ... * step[K-1]"),
        ("operation", "normalization", "automatic"),
        ("operation", "state_shape", "[B, 8]"),
        ("tolerances", "float64_oracle_atol", 1e-8),
        ("claims", "current_level", 0),
        ("claims", "performance_eligible", True),
        ("claims", "stable_benchmark", True),
        ("dependencies", "rqm_entanglement_runtime", True),
    ],
)
def test_preregistration_rejects_contract_and_claim_drift(
    section: str, field: str, value: object
) -> None:
    manifest = copy.deepcopy(load_entanglement_preregistration())
    manifest[section][field] = value

    with pytest.raises(EntanglementPreregistrationError):
        validate_entanglement_preregistration(manifest)


def test_preregistration_rejects_case_removal() -> None:
    manifest = copy.deepcopy(load_entanglement_preregistration())
    manifest["cases"].pop()

    with pytest.raises(EntanglementPreregistrationError):
        validate_entanglement_preregistration(manifest)


@pytest.mark.parametrize(
    "field",
    ["hardware_release", "performance_cases", "raw_sessions", "plots", "throughput", "device"],
)
def test_preregistration_rejects_hardware_and_performance_fields(field: str) -> None:
    manifest = copy.deepcopy(load_entanglement_preregistration())
    manifest[field] = []

    with pytest.raises(EntanglementPreregistrationError):
        validate_entanglement_preregistration(manifest)
