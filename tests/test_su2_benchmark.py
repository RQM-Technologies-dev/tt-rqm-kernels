from __future__ import annotations

import copy

import pytest

from tt_rqm_kernels.su2_benchmark import (
    PreregistrationError,
    load_su2_preregistration,
    validate_su2_preregistration,
)


def test_committed_su2_preregistration_is_valid() -> None:
    manifest = load_su2_preregistration()
    assert manifest["claims"]["current_level"] is None
    assert manifest["status"] == "pre_hardware"


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("operation", "composition_order", "step[0] * ... * step[K-1]"),
        ("operation", "normalization", "automatic"),
        ("methodology", "device_count", 2),
        ("methodology", "samples", 9),
        ("tolerances", "hardware_atol", 1e-3),
        ("claims", "current_level", 1),
        ("claims", "stable_benchmark", True),
    ],
)
def test_preregistration_rejects_methodology_and_claim_drift(
    section: str, field: str, value: object
) -> None:
    manifest = copy.deepcopy(load_su2_preregistration())
    manifest[section][field] = value
    with pytest.raises(PreregistrationError):
        validate_su2_preregistration(manifest)


def test_preregistration_rejects_performance_case_cherry_picking() -> None:
    manifest = copy.deepcopy(load_su2_preregistration())
    manifest["performance_cases"].pop()
    with pytest.raises(PreregistrationError, match="performance matrix"):
        validate_su2_preregistration(manifest)


def test_preregistration_requires_all_nonclaims() -> None:
    manifest = copy.deepcopy(load_su2_preregistration())
    manifest["nonclaims"].remove("no_acceleration_claim")
    with pytest.raises(PreregistrationError, match="nonclaims"):
        validate_su2_preregistration(manifest)
