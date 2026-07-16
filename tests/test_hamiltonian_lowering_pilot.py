from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experimental.tt_metalium_hamiltonian_lowering.run_candidate import (
    source_bundle_sha256,
)
from tt_rqm_kernels.hamiltonian_lowering_benchmark import CASE_IDS
from tt_rqm_kernels.hamiltonian_lowering_pilot import (
    HamiltonianLoweringPilotError,
    PILOT_SCHEMA,
    PINNED_TT_METAL_COMMIT,
    SUITE_SCHEMA,
    frozen_case_input_hashes,
    validate_pilot_package,
)


def _package(tmp_path: Path) -> tuple[Path, dict[str, object], dict[str, object]]:
    identity = {
        "candidate_sha256": "a" * 64,
        "source_bundle_sha256": "b" * 64,
        "source_commit": "c" * 40,
        "tt_metal_commit": PINNED_TT_METAL_COMMIT,
        "device_arch": "wormhole_b0",
        "device_id": 0,
        "core_count": 1,
    }
    manifest: dict[str, object] = {
        "schema": PILOT_SCHEMA,
        "pilot_id": "test-pilot",
        "designated": False,
        "qualification_eligible": False,
        "claim_level": None,
        "stable_benchmark": False,
        "performance_eligible": False,
        "hardware_execution": True,
        "case_ids": list(CASE_IDS),
        "case_input_hashes": frozen_case_input_hashes(),
        "tt_metal_commit": PINNED_TT_METAL_COMMIT,
    }
    results = []
    for case_id in CASE_IDS:
        case_dir = tmp_path / "cases" / case_id
        case_dir.mkdir(parents=True)
        rotors = f"rotors-{case_id}".encode()
        phases = f"phases-{case_id}".encode()
        (case_dir / "rotors.bin").write_bytes(rotors)
        (case_dir / "phases.bin").write_bytes(phases)
        checksum = hashlib.sha256(rotors + phases).hexdigest()
        report = {
            "execution_label": "hardware",
            "input_hashes": frozen_case_input_hashes()[case_id],
            "correctness": {
                "passed": True,
                "failing_value_count": 0,
                "nonfinite_value_count": 0,
                "checksum": checksum,
            },
            "candidate_metrics": {"candidate_metadata": identity},
        }
        (case_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")
        results.append(
            {
                "case_id": case_id,
                "passed": True,
                "report": f"cases/{case_id}/report.json",
                "output_checksum": checksum,
            }
        )
    suite: dict[str, object] = {
        "schema": SUITE_SCHEMA,
        "pilot_id": "test-pilot",
        "case_ids": list(CASE_IDS),
        "results": results,
        "candidate_identity_consistent": True,
        "candidate_identity": identity,
        "suite_passed": True,
        "stable_benchmark": False,
        "performance_eligible": False,
        "claim_level": None,
    }
    (tmp_path / "candidate-metadata.json").write_text(json.dumps(identity), encoding="utf-8")
    (tmp_path / "pilot-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "suite-report.json").write_text(json.dumps(suite), encoding="utf-8")
    return tmp_path, manifest, suite


def _rewrite(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_source_bundle_hash_is_deterministic_and_bound_to_source(tmp_path: Path) -> None:
    (tmp_path / "a.cpp").write_text("a", encoding="utf-8")
    first = source_bundle_sha256(tmp_path)
    assert first == source_bundle_sha256(tmp_path)
    (tmp_path / "a.cpp").write_text("b", encoding="utf-8")
    assert source_bundle_sha256(tmp_path) != first


def test_frozen_case_order_and_input_hashes_are_complete() -> None:
    assert tuple(frozen_case_input_hashes()) == CASE_IDS
    assert all(
        set(value) == {"hamiltonians_sha256", "dt_sha256"}
        for value in frozen_case_input_hashes().values()
    )


def test_passing_non_designated_package_validates_offline(tmp_path: Path) -> None:
    root, _, _ = _package(tmp_path)
    assert validate_pilot_package(root) == {
        "package_valid": True,
        "pilot_passed": True,
        "case_count": 9,
    }


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("designated", True, "designated"),
        ("stable_benchmark", True, "stable_benchmark"),
        ("performance_eligible", True, "performance_eligible"),
        ("claim_level", 0, "claim_level"),
    ],
)
def test_pilot_manifest_rejects_claim_promotion(
    tmp_path: Path, key: str, value: object, message: str
) -> None:
    root, manifest, _ = _package(tmp_path)
    manifest[key] = value
    _rewrite(root / "pilot-manifest.json", manifest)
    with pytest.raises(HamiltonianLoweringPilotError, match=message):
        validate_pilot_package(root)


def test_pilot_rejects_missing_or_reordered_case(tmp_path: Path) -> None:
    root, _, suite = _package(tmp_path)
    suite["results"] = list(reversed(suite["results"]))
    _rewrite(root / "suite-report.json", suite)
    with pytest.raises(HamiltonianLoweringPilotError, match="missing or reordered"):
        validate_pilot_package(root)


def test_pilot_rejects_changed_candidate_or_tt_metal_identity(tmp_path: Path) -> None:
    root, _, _ = _package(tmp_path)
    report_path = root / "cases" / CASE_IDS[-1] / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["candidate_metrics"]["candidate_metadata"]["candidate_sha256"] = "d" * 64
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(HamiltonianLoweringPilotError, match="identity changed"):
        validate_pilot_package(root)

    root2, _, _ = _package(tmp_path / "tt")
    report_path = root2 / "cases" / CASE_IDS[0] / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["candidate_metrics"]["candidate_metadata"]["tt_metal_commit"] = "e" * 40
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(HamiltonianLoweringPilotError, match="TT-Metal"):
        validate_pilot_package(root2)
