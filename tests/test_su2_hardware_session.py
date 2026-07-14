from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tt_rqm_kernels.benchmark_integrity import IntegrityError
from tt_rqm_kernels.su2_hardware_session import collect_su2_session


ROOT = Path(__file__).resolve().parents[1]


def _git_repo(path: Path) -> None:
    path.mkdir()
    subprocess.run(["git", "init", "-q", path], check=True)
    subprocess.run(["git", "-C", path, "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", path, "config", "user.email", "test@example.com"], check=True)
    (path / "tracked.txt").write_text("clean\n")
    subprocess.run(["git", "-C", path, "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", path, "commit", "-qm", "initial"], check=True)


def _health_script(path: Path) -> str:
    devices = []
    for device_id in range(2):
        devices.append(
            {
                "board_info": {
                    "board_type": f"n300-{device_id}",
                    "board_id": "board",
                    "bus_id": str(device_id),
                    "dram_status": True,
                },
                "smbus_telem": {
                    "FAULTS": "0x0",
                    "THROTTLER": "0x0",
                    "BOOT_DATE": "boot",
                    "RT_SECONDS": "0x10",
                },
                "telemetry": {
                    "asic_temperature": "40.0",
                    "aiclk": "500",
                    "heartbeat": "10",
                },
                "limits": {"thm_limit": "75"},
            }
        )
    path.write_text(
        f"import json\nprint(json.dumps({devices!r} and {{'device_info': {devices!r}}}))\n"
    )
    return f"{sys.executable} {path}"


def _candidate(path: Path) -> str:
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(path.stat().st_mode | 0o111)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _report() -> dict[str, object]:
    report = json.loads(
        (ROOT / "reports/tt_hardware_su2_compose_first_comparison.json").read_text()
    )
    for result in report["results"]:
        result["input_hashes"] = {
            "rotors_sha256": "0" * 64,
            "phases_sha256": "1" * 64,
        }
    return report


def test_collector_writes_complete_hash_bound_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, metal = tmp_path / "repository", tmp_path / "tt-metal"
    _git_repo(repository)
    _git_repo(metal)
    candidate = tmp_path / "candidate"
    candidate_hash = _candidate(candidate)
    metal_commit = subprocess.run(
        ["git", "-C", metal, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    def fake_run(**kwargs):
        kwargs["process_capture"].update({"stdout": "candidate out\n", "stderr": ""})
        return _report()

    monkeypatch.setattr("tt_rqm_kernels.su2_hardware_session.run_su2_compose", fake_run)
    directory = collect_su2_session(
        session_dir=tmp_path / "session",
        session_id="session-2",
        command=str(candidate),
        methodology_note="designated",
        repository_root=repository,
        tt_metal_root=metal,
        expected_candidate_sha256=candidate_hash,
        expected_execution_source_commit="3238299a9eea2a44dccd6826a947cac3266dd2f7",
        expected_tt_metal_commit=metal_commit,
        tt_smi_command=_health_script(tmp_path / "health.py"),
    )

    manifest = json.loads((directory / "session-manifest.json").read_text())
    assert manifest["collection_status"] == "passed"
    assert manifest["no_discarded_performance_runs"] is True
    assert manifest["all_expected_paired_samples_retained"] is True
    roles = {artifact["role"] for artifact in manifest["artifacts"]}
    assert roles == {
        "candidate-identity",
        "candidate-stderr",
        "candidate-stdout",
        "environment",
        "exact-command",
        "hardware-report",
        "hardware-report-summary",
        "input-hashes",
        "post-device-health",
        "pre-device-health",
    }
    for artifact in manifest["artifacts"]:
        path = directory / artifact["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]


def test_collector_preserves_failed_designated_session(tmp_path: Path) -> None:
    repository, metal = tmp_path / "repository", tmp_path / "tt-metal"
    _git_repo(repository)
    _git_repo(metal)
    (repository / "tracked.txt").write_text("dirty\n")
    candidate = tmp_path / "candidate"
    candidate_hash = _candidate(candidate)
    metal_commit = subprocess.run(
        ["git", "-C", metal, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    directory = tmp_path / "failed-session"

    with pytest.raises(IntegrityError, match="evidence preserved"):
        collect_su2_session(
            session_dir=directory,
            session_id="failed",
            command=str(candidate),
            methodology_note="designated",
            repository_root=repository,
            tt_metal_root=metal,
            expected_candidate_sha256=candidate_hash,
            expected_execution_source_commit="3238299a9eea2a44dccd6826a947cac3266dd2f7",
            expected_tt_metal_commit=metal_commit,
            tt_smi_command=_health_script(tmp_path / "health.py"),
        )

    manifest = json.loads((directory / "session-manifest.json").read_text())
    assert manifest["collection_status"] == "failed"
    assert manifest["source_trees_clean"] is False
    assert "source tree is dirty" in manifest["failure"]
    assert (directory / "stdout.txt").is_file()
    assert (directory / "stderr.txt").is_file()
