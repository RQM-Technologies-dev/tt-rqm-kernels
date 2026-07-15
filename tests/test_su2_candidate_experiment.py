from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from tt_rqm_kernels.su2_candidate_experiment import (
    DEFAULT_MANIFEST,
    SU2CandidateExperimentError,
    load_manifest,
    validate_candidate_experiment,
    validate_manifest,
)


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_experiment(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    manifest = copy.deepcopy(load_manifest(ROOT / DEFAULT_MANIFEST))
    for package in manifest["packages"]:
        source = ROOT / package["path"]
        destination = tmp_path / package["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination)
    manifest_path = tmp_path / DEFAULT_MANIFEST
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path, manifest


def _refresh_inventory(
    tmp_path: Path, manifest: dict[str, object], role: str, relative: str
) -> None:
    package = next(value for value in manifest["packages"] if value["role"] == role)
    directory = tmp_path / package["path"]
    inventory = directory / "artifacts.sha256"
    lines = []
    for line in inventory.read_text().splitlines():
        _, path = line.split("  ./", 1)
        digest = _sha256(directory / path) if path == relative else line.split()[0]
        lines.append(f"{digest}  ./{path}")
    inventory.write_text("\n".join(lines) + "\n")
    package["inventory_sha256"] = _sha256(inventory)


def test_retained_candidate_experiment_validates() -> None:
    manifest = validate_candidate_experiment(ROOT / DEFAULT_MANIFEST, repo_root=ROOT)
    assert manifest["candidate"]["sha256"].startswith("54b91b")
    assert manifest["claim"] == {
        "designated_stability_session": False,
        "performance_eligible": True,
        "stable_benchmark": False,
    }


def test_candidate_experiment_rejects_stability_promotion() -> None:
    manifest = copy.deepcopy(load_manifest(ROOT / DEFAULT_MANIFEST))
    manifest["claim"]["stable_benchmark"] = True
    with pytest.raises(SU2CandidateExperimentError, match="remain non-stable"):
        validate_manifest(manifest, repo_root=ROOT)


def test_candidate_experiment_rejects_inventory_hash_tampering(tmp_path: Path) -> None:
    manifest_path, manifest = _copy_experiment(tmp_path)
    package = next(value for value in manifest["packages"] if value["role"] == "performance")
    inventory = tmp_path / package["path"] / "artifacts.sha256"
    inventory.write_text(inventory.read_text().replace("0", "1", 1))
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with pytest.raises(SU2CandidateExperimentError, match="inventory SHA-256 mismatch"):
        validate_candidate_experiment(manifest_path.relative_to(tmp_path), repo_root=tmp_path)


def test_candidate_experiment_rejects_unlisted_file(tmp_path: Path) -> None:
    manifest_path, _ = _copy_experiment(tmp_path)
    package = (
        tmp_path
        / "benchmarks/raw/su2-compose/2026-07-15-n300-device0-candidate-54b91b-experiment-1"
    )
    (package / "unlisted.txt").write_text("not in inventory\n")
    with pytest.raises(SU2CandidateExperimentError, match="inventory coverage mismatch"):
        validate_candidate_experiment(manifest_path.relative_to(tmp_path), repo_root=tmp_path)


def test_candidate_experiment_rejects_candidate_binary_tampering(tmp_path: Path) -> None:
    manifest_path, manifest = _copy_experiment(tmp_path)
    package = next(value for value in manifest["packages"] if value["role"] == "performance")
    binary = tmp_path / package["path"] / package["candidate_binary"]
    binary.write_bytes(binary.read_bytes() + b"tampered")
    _refresh_inventory(tmp_path, manifest, "performance", binary.name)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with pytest.raises(SU2CandidateExperimentError, match="candidate binary mismatch"):
        validate_candidate_experiment(manifest_path.relative_to(tmp_path), repo_root=tmp_path)


def test_candidate_experiment_rejects_dirty_tree_record(tmp_path: Path) -> None:
    manifest_path, manifest = _copy_experiment(tmp_path)
    package = next(value for value in manifest["packages"] if value["role"] == "performance")
    status = tmp_path / package["path"] / "execution-source-status.txt"
    status.write_text(" M candidate.cpp\n")
    _refresh_inventory(tmp_path, manifest, "performance", status.name)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with pytest.raises(SU2CandidateExperimentError, match="dirty tree"):
        validate_candidate_experiment(manifest_path.relative_to(tmp_path), repo_root=tmp_path)


def test_candidate_experiment_rejects_unhealthy_device_snapshot(tmp_path: Path) -> None:
    manifest_path, manifest = _copy_experiment(tmp_path)
    package = next(value for value in manifest["packages"] if value["role"] == "performance")
    health = tmp_path / package["path"] / "post-performance-device-health.txt"
    health.write_text(health.read_text().replace('"FAULTS": "0x0"', '"FAULTS": "0x1"', 1))
    _refresh_inventory(tmp_path, manifest, "performance", health.name)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with pytest.raises((SU2CandidateExperimentError, ValueError), match="fault|Fault|healthy"):
        validate_candidate_experiment(manifest_path.relative_to(tmp_path), repo_root=tmp_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda report: report["results"].pop(), "case order mismatch"),
        (lambda report: report["results"][0].update({"samples": 9}), "sample count mismatch"),
        (
            lambda report: report["results"][0]["raw_candidate_timings_s"].update(
                {"paired_order": ["unfused_first"] * 10}
            ),
            "paired timing order mismatch",
        ),
        (
            lambda report: report["results"][0]["fused"]["correctness"].update(
                {"nonfinite_values": 1}
            ),
            "nonfinite result",
        ),
    ],
)
def test_candidate_experiment_rejects_report_tampering(
    tmp_path: Path, mutation, message: str
) -> None:
    manifest_path, manifest = _copy_experiment(tmp_path)
    package = next(value for value in manifest["packages"] if value["role"] == "performance")
    report_path = tmp_path / package["path"] / "performance.json"
    report = json.loads(report_path.read_text())
    mutation(report)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _refresh_inventory(tmp_path, manifest, "performance", report_path.name)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with pytest.raises(SU2CandidateExperimentError, match=message):
        validate_candidate_experiment(manifest_path.relative_to(tmp_path), repo_root=tmp_path)
