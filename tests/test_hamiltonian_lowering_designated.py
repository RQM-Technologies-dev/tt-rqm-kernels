from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

import tt_rqm_kernels.hamiltonian_lowering_designated as designated
from tt_rqm_kernels.hamiltonian_lowering_candidate import HamiltonianLoweringCandidateError

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / designated.MANIFEST_PATH
CLEAN_PILOT = (
    ROOT
    / "benchmarks/pilots/hamiltonian-lowering-h2a"
    / "h2a-clean-reproduction-20260716/nine-case-pilot"
)


def test_frozen_designated_contract_and_inputs_validate() -> None:
    result = designated.validate_designated_manifest(MANIFEST, ROOT)
    assert result["designated_manifest_valid"] is True
    assert result["frozen_inputs_valid"] is True
    assert result["case_count"] == 9
    readiness = designated.contract_readiness(MANIFEST, ROOT)
    assert readiness["qualifier_ready"] is True
    assert readiness["designated_session_present"] is False
    assert readiness["qualification_passed"] is None
    assert readiness["claim_level"] is None


def test_frozen_inputs_reproduce_byte_for_byte(tmp_path: Path) -> None:
    first = designated.validate_frozen_inputs(ROOT / designated.INPUT_ROOT)
    generated = tmp_path / "inputs"
    designated.freeze_inputs(generated)
    second = designated.validate_frozen_inputs(generated)
    assert second["input_manifest_sha256"] == first["input_manifest_sha256"]
    for relative in sorted(
        path.relative_to(generated) for path in generated.rglob("*") if path.is_file()
    ):
        assert (generated / relative).read_bytes() == (
            ROOT / designated.INPUT_ROOT / relative
        ).read_bytes()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (
            "candidate_binary_sha256",
            "433e74b827d2cf9a7a790a6c9d7bb3917fc1fed3915ec384de0486cdc014d306",
        ),
        (
            "source_bundle_sha256",
            "7fb65217e05139bf035952ebeb34602d49e5f1772b8dec4c336b7a296e1fba2f",
        ),
        ("status", "collected"),
        ("claim_level", 0),
        ("stable_benchmark", True),
        ("performance_eligible", True),
        ("collection_started", True),
    ],
)
def test_designated_contract_rejects_promotion_or_development_identity(
    tmp_path: Path, field: str, value: object
) -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    payload[field] = value
    changed = tmp_path / "manifest.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(designated.HamiltonianLoweringDesignatedError):
        designated.validate_designated_manifest(changed, ROOT)


def test_frozen_contract_remains_immutable_after_separate_release() -> None:
    assert (ROOT / "benchmarks/manifests/wormhole-hamiltonian-lowering.json").is_file()
    assert (ROOT / "benchmarks/raw/hamiltonian-lowering-h2a").is_dir()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["status"] == "frozen_not_collected"
    assert manifest["completed_session_id"] is None


def test_device_health_requires_healthy_n300_device_zero() -> None:
    payload = {
        "device_info": [
            {
                "board_info": {
                    "board_type": "n300 L",
                    "dram_status": True,
                    "pcie_width": "16",
                },
                "smbus_telem": {"FAULTS": "0x0", "THROTTLER": "0x0"},
            }
        ]
    }
    assert designated.validate_device_health(payload) == {
        "device_id": 0,
        "board_type": "n300 L",
        "dram_status": True,
        "pcie_width": "16",
        "faults": "0x0",
        "throttler": "0x0",
    }
    payload["device_info"][0]["smbus_telem"]["FAULTS"] = "0x1"
    with pytest.raises(designated.HamiltonianLoweringDesignatedError, match="faults"):
        designated.validate_device_health(payload)


def test_collector_retains_all_failed_cases_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(designated, "dry_run_preflight", lambda **_: {"dry_run_passed": True})

    def fail(*args: object, **kwargs: object) -> object:
        raise HamiltonianLoweringCandidateError("retained failure")

    output = tmp_path / "session"
    session = designated.collect_designated_session(
        manifest_path=MANIFEST,
        governance_root=ROOT,
        source_repo=tmp_path / "source",
        tt_metal_root=tmp_path / "tt-metal",
        candidate_binary=tmp_path / "candidate",
        output_root=output,
        session_id="future-test-session",
        runner=fail,
    )
    assert session["collection_completed"] is True
    assert session["all_cases_passed"] is False
    assert [item["case_id"] for item in session["results"]] == list(designated.CASE_IDS)
    assert all(item["attempt"] == 1 and item["passed"] is False for item in session["results"])
    assert session["retries"] == 0
    assert session["replacement_results"] == 0
    assert all(
        (output / "cases" / case_id / "error.txt").is_file() for case_id in designated.CASE_IDS
    )


def test_future_session_qualifier_accepts_complete_hash_bound_fixture(tmp_path: Path) -> None:
    session_root = tmp_path / "session"
    shutil.copytree(CLEAN_PILOT / "cases", session_root / "cases")
    clean_suite = json.loads((CLEAN_PILOT / "suite-report.json").read_text(encoding="utf-8"))
    results = [
        {
            "case_id": item["case_id"],
            "attempt": 1,
            "passed": True,
            "report": item["report"],
            "checksum": item["output_checksum"],
        }
        for item in clean_suite["results"]
    ]
    session = {
        "schema": designated.SESSION_SCHEMA,
        "session_id": "future-test-session",
        "designated": True,
        "target_claim_level": 0,
        "claim_level": None,
        "collection_started": True,
        "collection_completed": True,
        "manifest_sha256": designated.sha256_file(MANIFEST),
        "candidate_binary_sha256": designated.CANDIDATE_SHA256,
        "repository_commit": designated.IMPLEMENTATION_COMMIT,
        "source_bundle_sha256": designated.SOURCE_BUNDLE_SHA256,
        "tt_metal_commit": designated.TT_METAL_COMMIT,
        "device_id": 0,
        "device_count": 1,
        "core_count": 1,
        "case_order": list(designated.CASE_IDS),
        "attempts_per_case": 1,
        "retries": 0,
        "replacement_results": 0,
        "stable_benchmark": False,
        "performance_eligible": False,
        "results": results,
    }
    (session_root / "session-manifest.json").write_text(
        json.dumps(session, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result = designated.qualify_session(session_root, MANIFEST, ROOT)
    assert result == {
        "schema": designated.QUALIFICATION_SCHEMA,
        "qualification_passed": True,
        "target_claim_level": 0,
        "claim_level": None,
        "stable_benchmark": False,
        "performance_eligible": False,
        "release_created": False,
    }


def test_qualifier_rejects_retry_or_session_promotion(tmp_path: Path) -> None:
    session_root = tmp_path / "session"
    session_root.mkdir()
    (session_root / "session-manifest.json").write_text(
        json.dumps({"schema": designated.SESSION_SCHEMA, "retries": 1}), encoding="utf-8"
    )
    with pytest.raises(designated.HamiltonianLoweringDesignatedError):
        designated.qualify_session(session_root, MANIFEST, ROOT)
