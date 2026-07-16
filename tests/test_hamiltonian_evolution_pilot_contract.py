from __future__ import annotations

import json
from pathlib import Path

from tt_rqm_kernels.hamiltonian_evolution_pilot_contract import (
    CASE_ORDER,
    DEFAULT_MANIFEST,
    validate_pilot_contract,
)
from tt_rqm_kernels.hamiltonian_evolution_source_identity import validate_source_manifest

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_contract_binds_identity_domain_and_one_attempt_policy() -> None:
    contract = validate_pilot_contract(ROOT / DEFAULT_MANIFEST, ROOT)
    assert contract["status"] == "pilot_frozen_before_first_n300_run"
    assert contract["non_designated"] is True
    assert contract["claim_level"] is None
    assert contract["stable_benchmark"] is False
    assert contract["performance_eligible"] is False
    assert contract["case_order"] == list(CASE_ORDER)
    assert contract["attempt_policy"] == {
        "attempts_per_case": 1,
        "retries": 0,
        "replacement": "forbidden",
        "first_result_retained_regardless_of_outcome": True,
    }
    assert contract["numerical_domain"]["rotor_angle_limit"] == 1024.0
    assert contract["numerical_domain"]["phase_angle_limit"] == 8192.0
    assert contract["cases"][-1]["case_id"] == "large_angle_short_chain"
    assert contract["cases"][-1]["role"] == "stress_diagnostic"


def test_source_manifest_and_large_angle_diagnostic_are_hash_bound() -> None:
    source = validate_source_manifest(
        ROOT / "benchmarks/manifests/hamiltonian-evolution-h2b-source-manifest.json",
        ROOT,
    )
    assert source["source_scope_clean"] is True
    assert source["file_count"] == 23
    diagnostic = json.loads((ROOT / "reports/h2b_large_angle_diagnostic.json").read_text())
    assert diagnostic["diagnosis"]["acceptance_path"] == "B_formally_bounded_operating_domain"
    assert diagnostic["sweep"]["case_count"] == 166
