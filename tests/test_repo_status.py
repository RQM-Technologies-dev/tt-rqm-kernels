from __future__ import annotations

import json
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys

from scripts.repo_status import (
    _entanglement_foundation_status,
    _entanglement_hardware_status,
    _su2_comparison_status,
    _su2_conformance_status,
    _su2_foundation_status,
    _su2_stability_status,
)


ROOT = Path(__file__).resolve().parents[1]


def test_repo_status_json_reports_current_gaps() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/repo_status.py", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["schema"] == "tt-rqm-repo-status.v1"
    statuses = {item["name"]: item["status"] for item in payload["items"]}
    assert statuses["CPU/PyTorch reference"] == "implemented"
    assert statuses["StructuredBench smoke"] == "implemented"
    assert statuses["external-qmul harness"] == "implemented"
    assert statuses["TT-Metalium candidate"] == "experimental source candidate present"
    assert statuses["tt-emule candidate"] == "emulation report present"
    assert statuses["hardware report"] == "hardware conformance report present"
    assert statuses["Stage B hardware report"] == "first hardware sample present"
    assert statuses["Persistent Stage B hardware report"] == "stable one-device performance present"
    assert statuses["SU2ComposeBench reference foundation"] == "implemented reference"
    assert statuses["SU2ComposeBench N300 conformance"] == "hardware conformance present"
    assert statuses["SU2ComposeBench first comparison"] == "qualified first comparison present"
    assert statuses["SU2ComposeBench stability"] == "not established"
    assert statuses["EntanglementDynamicsBench reference foundation"] == "implemented reference"
    assert statuses["EntanglementDynamicsBench hardware"] == "not implemented"


def test_repo_status_text_is_maintainer_scannable() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/repo_status.py"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "CPU/PyTorch reference: implemented" in completed.stdout
    assert "TT-Metalium candidate: experimental source candidate present" in completed.stdout
    assert "tt-emule candidate: emulation report present" in completed.stdout
    assert "hardware report: hardware conformance report present" in completed.stdout
    assert "Stage B hardware report: first hardware sample present" in completed.stdout
    assert (
        "Persistent Stage B hardware report: stable one-device performance present"
        in completed.stdout
    )
    assert "SU2ComposeBench reference foundation: implemented reference" in completed.stdout
    assert "SU2ComposeBench N300 conformance: hardware conformance present" in completed.stdout
    assert (
        "SU2ComposeBench first comparison: qualified first comparison present" in completed.stdout
    )
    assert "SU2ComposeBench stability: not established" in completed.stdout
    assert (
        "EntanglementDynamicsBench reference foundation: implemented reference" in completed.stdout
    )
    assert "EntanglementDynamicsBench hardware: not implemented" in completed.stdout
    assert "not performance-eligible" in completed.stdout
    assert "not an acceleration claim" in completed.stdout


def test_su2_status_fails_clearly_when_evidence_is_absent(tmp_path: Path) -> None:
    assert _su2_foundation_status(tmp_path)[0] == "not implemented"
    assert _su2_conformance_status(tmp_path)[0] == "not implemented"
    assert _su2_comparison_status(tmp_path)[0] == "not implemented"
    assert _su2_stability_status(tmp_path)[0] == "not established"


def test_entanglement_status_fails_clearly_when_foundation_is_absent(tmp_path: Path) -> None:
    assert _entanglement_foundation_status(tmp_path)[0] == "not implemented"
    assert _entanglement_hardware_status(tmp_path)[0] == "not implemented"


def test_entanglement_status_rejects_preregistration_claim_escalation(
    tmp_path: Path,
) -> None:
    for relative in (
        "benchmarks/manifests/entanglement-dynamics-preregistration.json",
        "tt_rqm_kernels/hamiltonian/two_qubit.py",
        "tt_rqm_kernels/hamiltonian/two_qubit_metrics.py",
        "tt_rqm_kernels/hamiltonian/__init__.py",
    ):
        _copy_path(relative, tmp_path)
    path = tmp_path / "benchmarks/manifests/entanglement-dynamics-preregistration.json"
    payload = json.loads(path.read_text())
    payload["claims"]["current_level"] = 0
    path.write_text(json.dumps(payload))

    assert _entanglement_foundation_status(tmp_path)[0] == "invalid reference foundation"


def test_entanglement_status_rejects_unplanned_hardware_evidence(tmp_path: Path) -> None:
    report = tmp_path / "reports/tt_hardware_entanglement_dynamics.json"
    report.parent.mkdir(parents=True)
    report.write_text("{}")

    assert _entanglement_hardware_status(tmp_path)[0] == "unexpected evidence present"


def test_su2_status_rejects_malformed_foundation(tmp_path: Path) -> None:
    _copy_path("benchmarks/manifests/su2-compose-preregistration.json", tmp_path)
    for relative in (
        "tt_rqm_kernels/hamiltonian/__init__.py",
        "tt_rqm_kernels/hamiltonian/su2_lowering.py",
        "tt_rqm_kernels/hamiltonian/su2_compose.py",
        "tt_rqm_kernels/hamiltonian/su2_reference.py",
    ):
        _copy_path(relative, tmp_path)
    path = tmp_path / "benchmarks/manifests/su2-compose-preregistration.json"
    payload = json.loads(path.read_text())
    payload["operation"]["composition_order"] = "step[0] * ... * step[K-1]"
    path.write_text(json.dumps(payload))
    assert _su2_foundation_status(tmp_path)[0] == "invalid reference foundation"


def test_su2_conformance_rejects_wrong_device_scope(tmp_path: Path) -> None:
    _copy_conformance_release(tmp_path)
    report_path = tmp_path / "reports/tt_hardware_su2_compose_conformance.json"
    report = json.loads(report_path.read_text())
    report["lifecycle"]["device_id"] = 1
    report_path.write_text(json.dumps(report))
    _rehash_manifest_artifact(
        tmp_path / "benchmarks/manifests/su2-compose-conformance.json",
        "reports/tt_hardware_su2_compose_conformance.json",
        report_path,
    )
    assert _su2_conformance_status(tmp_path)[0] == "invalid conformance evidence"


def test_su2_comparison_rejects_stable_or_session_promotion(tmp_path: Path) -> None:
    _copy_performance_release(tmp_path)
    manifest_path = tmp_path / "benchmarks/manifests/wormhole-su2-compose.json"
    release = json.loads(manifest_path.read_text())
    release["claim"].update({"level": 2, "public_session_count": 3, "stable_benchmark": True})
    manifest_path.write_text(json.dumps(release))
    assert _su2_comparison_status(tmp_path)[0] == "invalid comparison evidence"
    assert _su2_stability_status(tmp_path)[0] == "invalid stability status"


def test_su2_comparison_rejects_wrong_device_scope(tmp_path: Path) -> None:
    _copy_performance_release(tmp_path)
    manifest_path = tmp_path / "benchmarks/manifests/wormhole-su2-compose.json"
    release = json.loads(manifest_path.read_text())
    report_path = tmp_path / release["primary_report"]
    report = json.loads(report_path.read_text())
    report["lifecycle"]["device_id"] = 1
    report_path.write_text(json.dumps(report))
    _rehash_manifest_artifact(manifest_path, release["primary_report"], report_path)
    assert _su2_comparison_status(tmp_path)[0] == "invalid comparison evidence"


def _copy_path(relative: str, destination: Path) -> None:
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / relative, target)


def _copy_conformance_release(destination: Path) -> None:
    manifest = json.loads((ROOT / "benchmarks/manifests/su2-compose-conformance.json").read_text())
    _copy_path("benchmarks/manifests/su2-compose-conformance.json", destination)
    for artifact in manifest["artifacts"]:
        _copy_path(artifact["path"], destination)
    _copy_performance_release(destination)


def _copy_performance_release(destination: Path) -> None:
    manifest = json.loads((ROOT / "benchmarks/manifests/wormhole-su2-compose.json").read_text())
    _copy_path("benchmarks/manifests/wormhole-su2-compose.json", destination)
    for artifact in manifest["artifacts"]:
        _copy_path(artifact["path"], destination)


def _rehash_manifest_artifact(manifest_path: Path, relative: str, artifact_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text())
    for artifact in manifest["artifacts"]:
        if artifact["path"] == relative:
            artifact["sha256"] = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            break
    manifest_path.write_text(json.dumps(manifest))
