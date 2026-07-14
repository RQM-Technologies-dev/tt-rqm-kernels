from __future__ import annotations

import json
from pathlib import Path

import pytest

from tt_rqm_kernels.benchmark_integrity import IntegrityError
from tt_rqm_kernels.hardware_session import compare_device_health, sha256_file, validate_device_health
from tt_rqm_kernels.qmul_stability import qualify_stability


ROOT = Path(__file__).resolve().parents[1]


def _health(*, aiclk: int = 500, faults: str = "0x0") -> str:
    devices = []
    for device_id in range(2):
        devices.append({
            "board_info": {"board_type": f"n300-{device_id}", "board_id": "board", "bus_id": str(device_id), "dram_status": True},
            "smbus_telem": {"FAULTS": faults, "THROTTLER": "0x0", "BOOT_DATE": "boot", "RT_SECONDS": "0x10"},
            "telemetry": {"asic_temperature": "40.0", "aiclk": str(aiclk), "heartbeat": "10"},
            "limits": {"thm_limit": "75"},
        })
    return json.dumps({"device_info": devices})


def test_health_validation_requires_both_clean_devices_and_fixed_clock() -> None:
    value = validate_device_health(_health(), device_id=1)
    assert value["visible_device_count"] == 2
    assert value["selected_device_id"] == 1
    compare_device_health(_health(), _health(), device_id=0)
    with pytest.raises(IntegrityError, match="AICLK changed"):
        compare_device_health(_health(), _health(aiclk=800), device_id=0)
    with pytest.raises(IntegrityError, match="hardware faults"):
        validate_device_health(_health(faults="0x1"), device_id=0)


def _session(tmp_path: Path, session_id: str, *, noisy: bool = False) -> Path:
    directory = tmp_path / session_id
    directory.mkdir()
    report = json.loads((ROOT / "reports/tt_hardware_qmul_stage_b_persistent_performance.json").read_text())
    if noisy:
        timing = report["results"][1]["timing"]["device_s"]
        timing["p95"] = timing["median"] * 1.2
    report_path = directory / "report.json"
    report_path.write_text(json.dumps(report, sort_keys=True) + "\n")
    manifest = {
        "schema": "tt-rqm-benchmark-session.v2",
        "session_id": session_id,
        "collection_status": "passed",
        "cold_start_host_session": True,
        "benchmark_stage": "performance",
        "stable_benchmark": False,
        "device_count": 1,
        "device_id": 0,
        "candidate_sha256": "179a5cc3e6b146a1e8c61e61ab9ab173bbc543f88181b91c8621a7e959c98ce5",
        "execution_source_commit": "3ae68815e8ac025e49f09d3797dbbac2f77245b3",
        "tt_metal_commit": "dd2849b5bc6b7a5d38a9eafbeba31ef8d530f8d4",
        "artifacts": [{"path": "report.json", "role": "hardware-report", "sha256": sha256_file(report_path)}],
    }
    path = directory / "session-manifest.json"
    path.write_text(json.dumps(manifest) + "\n")
    return path


def test_stability_requires_three_distinct_cold_start_sessions(tmp_path: Path) -> None:
    one = _session(tmp_path, "one")
    two = _session(tmp_path, "two")
    assert qualify_stability([one, two])["stable_benchmark"] is False
    three = _session(tmp_path, "three")
    result = qualify_stability([one, two, three])
    assert result["stable_benchmark"] is True
    assert all(value["passed"] for value in result["sizes"])


def test_stability_preserves_threshold_breach_in_decision(tmp_path: Path) -> None:
    paths = [_session(tmp_path, "one"), _session(tmp_path, "two"), _session(tmp_path, "three", noisy=True)]
    result = qualify_stability(paths)
    assert result["stable_benchmark"] is False
    size = next(value for value in result["sizes"] if value["items"] == 65536)
    assert any("within-session dispersion" in value for value in size["rejected_gates"])
