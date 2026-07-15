from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Callable

import pytest

from tt_rqm_kernels.benchmark_integrity import IntegrityError
from tt_rqm_kernels.su2_stability import (
    load_stability_preregistration,
    qualify_stability,
    sha256_file,
    validate_stability_preregistration,
    write_qualification,
)
from tt_rqm_kernels.su2_benchmark_release import (
    LEVEL2_NONCLAIMS,
    generate_release,
    validate_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
ReportMutation = Callable[[dict[str, object]], None]


def test_v2_stability_preregistration_is_frozen_before_session_one() -> None:
    preregistration = load_stability_preregistration(
        Path("benchmarks/manifests/su2-compose-stability-preregistration-v2.json"),
        repo_root=ROOT,
    )
    assert preregistration["status"] == "frozen_before_designated_session_1"
    assert preregistration["calibration_experiment"]["designated_stability_session"] is False
    assert preregistration["candidate"]["sha256"].startswith("54b91b")
    assert len(preregistration["inputs"]) == 8


def _health() -> str:
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
    return json.dumps({"device_info": devices})


def _foundation(tmp_path: Path) -> Path:
    preregistration = tmp_path / "benchmarks/manifests/su2-compose-stability-preregistration.json"
    preregistration.parent.mkdir(parents=True)
    shutil.copy2(
        ROOT / "benchmarks/manifests/su2-compose-stability-preregistration.json",
        preregistration,
    )
    first_report = tmp_path / "reports/tt_hardware_su2_compose_first_comparison.json"
    first_report.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "reports/tt_hardware_su2_compose_first_comparison.json", first_report)
    return preregistration


def _session(
    tmp_path: Path,
    session_id: str,
    *,
    mutate_report: ReportMutation | None = None,
    no_discarded: bool = True,
    stable: bool = False,
) -> Path:
    directory = tmp_path / "sessions" / session_id
    directory.mkdir(parents=True)
    report = json.loads(
        (ROOT / "reports/tt_hardware_su2_compose_first_comparison.json").read_text()
    )
    for result in report["results"]:
        result["input_hashes"] = {
            "rotors_sha256": "0" * 64,
            "phases_sha256": "1" * 64,
        }
    if mutate_report is not None:
        mutate_report(report)
    report_path = directory / "report.json"
    report_path.write_text(json.dumps(report, sort_keys=True) + "\n")
    input_hashes = {
        "schema": "tt-rqm-su2-compose-input-hashes.v1",
        "seed": 0,
        "cases": [
            {
                "B": result["B"],
                "K": result["K"],
                "case_id": result["case_id"],
                **result["input_hashes"],
            }
            for result in report["results"]
        ],
    }
    files = {
        "candidate.sha256": "d8237f2e  candidate\n",
        "command.txt": "/candidate\n",
        "environment.json": json.dumps({"repository": {"status": ""}, "tt_metal": {"status": ""}}),
        "input-hashes.json": json.dumps(input_hashes),
        "post-device-health.txt": _health(),
        "pre-device-health.txt": _health(),
        "report.md": "summary\n",
        "stderr.txt": "",
        "stdout.txt": "",
    }
    for name, value in files.items():
        (directory / name).write_text(value)
    roles = {
        "candidate.sha256": "candidate-identity",
        "command.txt": "exact-command",
        "environment.json": "environment",
        "input-hashes.json": "input-hashes",
        "post-device-health.txt": "post-device-health",
        "pre-device-health.txt": "pre-device-health",
        "report.json": "hardware-report",
        "report.md": "hardware-report-summary",
        "stderr.txt": "candidate-stderr",
        "stdout.txt": "candidate-stdout",
    }
    manifest = {
        "schema": "tt-rqm-su2-compose-session.v2",
        "session_id": session_id,
        "collection_status": "passed",
        "cold_start_host_session": True,
        "no_discarded_performance_runs": no_discarded,
        "stable_benchmark": stable,
        "device_count": 1,
        "device_id": 0,
        "candidate_sha256": "d8237f2e5b05885167085d87a0400daf8b5feb0318d906285a1d263035294441",
        "execution_source_commit": "3238299a9eea2a44dccd6826a947cac3266dd2f7",
        "tt_metal_commit": "dd2849b5bc6b7a5d38a9eafbeba31ef8d530f8d4",
        "source_trees_clean": True,
        "case_order": [[result["B"], result["K"]] for result in report["results"]],
        "all_expected_paired_samples_retained": True,
        "lifecycle": report["lifecycle"],
        "artifacts": [
            {
                "path": name,
                "role": role,
                "sha256": sha256_file(directory / name),
            }
            for name, role in roles.items()
        ],
    }
    manifest_path = directory / "session-manifest.json"
    manifest_path.write_text(json.dumps(manifest) + "\n")
    return manifest_path


def _qualify(tmp_path: Path, manifests: list[Path]) -> dict[str, object]:
    preregistration = _foundation(tmp_path)
    return qualify_stability(
        manifests,
        preregistration_path=preregistration.relative_to(tmp_path),
        repo_root=tmp_path,
    )


def test_three_complete_sessions_qualify_deterministically(tmp_path: Path) -> None:
    manifests = [_session(tmp_path, value) for value in ("one", "two", "three")]
    first = _qualify(tmp_path, manifests)
    second = qualify_stability(
        manifests,
        preregistration_path=Path(
            "benchmarks/manifests/su2-compose-stability-preregistration.json"
        ),
        repo_root=tmp_path,
    )

    assert first == second
    assert first["qualification_passed"] is True
    assert first["stable_benchmark"] is True
    assert all(case["passed"] for case in first["cases"])


def test_legacy_first_session_can_join_two_v2_sessions(tmp_path: Path) -> None:
    preregistration = _foundation(tmp_path)
    relative = Path("benchmarks/raw/su2-compose/2026-07-14-n300-device0-session-1/session.json")
    legacy = tmp_path / relative
    legacy.parent.mkdir(parents=True)
    shutil.copy2(ROOT / relative, legacy)
    payload = json.loads(legacy.read_text())
    for artifact in payload["artifacts"]:
        source = ROOT / artifact["path"]
        target = tmp_path / artifact["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    manifests = [
        legacy,
        _session(tmp_path, "two"),
        _session(tmp_path, "three"),
    ]

    result = qualify_stability(
        manifests,
        preregistration_path=preregistration.relative_to(tmp_path),
        repo_root=tmp_path,
    )
    assert result["qualification_passed"] is True
    assert result["sessions"][0]["legacy_session_manifest"] is True


def test_stability_preregistration_rejects_threshold_tampering() -> None:
    payload = json.loads(
        (ROOT / "benchmarks/manifests/su2-compose-stability-preregistration.json").read_text()
    )
    payload["cases"][0]["limits"]["fused"] = 0.5

    with pytest.raises(IntegrityError, match="not derived"):
        validate_stability_preregistration(payload, repo_root=ROOT)


def test_two_sessions_cannot_qualify(tmp_path: Path) -> None:
    result = _qualify(tmp_path, [_session(tmp_path, "one"), _session(tmp_path, "two")])
    assert result["qualification_passed"] is False
    assert any("exactly 3" in reason for reason in result["rejected_gates"])


def test_duplicate_session_ids_cannot_qualify(tmp_path: Path) -> None:
    manifests = [_session(tmp_path, value) for value in ("one", "two", "three")]
    payload = json.loads(manifests[-1].read_text())
    payload["session_id"] = "two"
    manifests[-1].write_text(json.dumps(payload))
    result = _qualify(tmp_path, manifests)

    assert result["qualification_passed"] is False
    assert "session IDs are not distinct" in result["rejected_gates"]


def test_changed_provenance_cannot_qualify(tmp_path: Path) -> None:
    def mutate(report: dict[str, object]) -> None:
        report["provenance"]["candidate"]["runtime_version"] = "changed"

    manifests = [
        _session(tmp_path, "one"),
        _session(tmp_path, "two"),
        _session(tmp_path, "three", mutate_report=mutate),
    ]
    result = _qualify(tmp_path, manifests)
    assert result["qualification_passed"] is False
    assert any("identity differs" in reason for reason in result["rejected_gates"])


def test_missing_case_cannot_qualify(tmp_path: Path) -> None:
    def mutate(report: dict[str, object]) -> None:
        report["results"].pop()

    manifests = [
        _session(tmp_path, "one"),
        _session(tmp_path, "two"),
        _session(tmp_path, "three", mutate_report=mutate),
    ]
    assert _qualify(tmp_path, manifests)["qualification_passed"] is False


def test_noisy_case_above_threshold_cannot_qualify(tmp_path: Path) -> None:
    def mutate(report: dict[str, object]) -> None:
        report["results"][2]["fused"]["timing_s"]["samples"][-1] *= 2.0

    manifests = [
        _session(tmp_path, "one"),
        _session(tmp_path, "two"),
        _session(tmp_path, "three", mutate_report=mutate),
    ]
    result = _qualify(tmp_path, manifests)
    assert result["qualification_passed"] is False
    assert any(
        "within-session dispersion" in reason
        for case in result["cases"]
        for reason in case["rejected_gates"]
    )


def test_correctness_failure_cannot_qualify(tmp_path: Path) -> None:
    def mutate(report: dict[str, object]) -> None:
        report["results"][0]["fused"]["correctness"]["failing_values"] = 1

    manifests = [
        _session(tmp_path, "one"),
        _session(tmp_path, "two"),
        _session(tmp_path, "three", mutate_report=mutate),
    ]
    assert _qualify(tmp_path, manifests)["qualification_passed"] is False


def test_discarded_run_and_manual_stability_flag_cannot_qualify(tmp_path: Path) -> None:
    manifests = [
        _session(tmp_path, "one"),
        _session(tmp_path, "two", no_discarded=False),
        _session(tmp_path, "three", stable=True),
    ]
    result = _qualify(tmp_path, manifests)
    assert result["qualification_passed"] is False
    assert any("discarded" in reason for reason in result["rejected_gates"])
    assert any("stable_benchmark=false" in reason for reason in result["rejected_gates"])


def test_level_two_release_requires_and_recomputes_qualification(tmp_path: Path) -> None:
    manifests = [_session(tmp_path, value) for value in ("one", "two", "three")]
    qualification = _qualify(tmp_path, manifests)
    qualification_path = tmp_path / "benchmarks/processed/su2-stability.json"
    write_qualification(qualification_path, qualification)

    supporting = {
        "benchmarks/manifests/su2-compose-preregistration.json": (
            ROOT / "benchmarks/manifests/su2-compose-preregistration.json"
        ),
        "reports/tt_hardware_su2_compose_architecture_audit.md": (
            ROOT / "reports/tt_hardware_su2_compose_architecture_audit.md"
        ),
        "reports/tt_hardware_su2_compose_timing_audit.md": (
            ROOT / "reports/tt_hardware_su2_compose_timing_audit.md"
        ),
    }
    for relative, source in supporting.items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    artifact_roles = {
        "benchmarks/manifests/su2-compose-preregistration.json": "preregistration",
        "benchmarks/manifests/su2-compose-stability-preregistration.json": "stability-preregistration",
        "benchmarks/processed/su2-stability.json": "stability-qualification",
        "reports/tt_hardware_su2_compose_architecture_audit.md": "pre-eligibility-architecture-audit",
        "reports/tt_hardware_su2_compose_timing_audit.md": "timing-audit",
    }
    sessions = []
    for manifest in manifests:
        manifest_relative = str(manifest.relative_to(tmp_path))
        report_relative = str((manifest.parent / "report.json").relative_to(tmp_path))
        artifact_roles[manifest_relative] = "session-manifest"
        artifact_roles[report_relative] = "session-performance-report"
        sessions.append(
            {
                "id": manifest.parent.name,
                "performance_report": report_relative,
                "session_manifest": manifest_relative,
            }
        )
    report = json.loads((manifests[0].parent / "report.json").read_text())
    candidate = report["provenance"]["candidate"]
    release = {
        "schema": "tt-rqm-su2-compose-release.v1",
        "benchmark_id": "wormhole-su2-compose-fp32",
        "primary_report": sessions[0]["performance_report"],
        "artifacts": [
            {
                "path": relative,
                "role": role,
                "sha256": sha256_file(tmp_path / relative),
            }
            for relative, role in artifact_roles.items()
        ],
        "provenance": {
            "candidate_sha256": candidate["candidate_sha256"],
            "repository_commit": candidate["repository_commit"],
            "tt_metal_commit": candidate["tt_metal_commit"],
        },
        "claim": {
            "level": 2,
            "name": "stable_one_device_performance",
            "public_session_count": 3,
            "stable_benchmark": True,
        },
        "sessions": sessions,
        "stability_qualification": "benchmarks/processed/su2-stability.json",
        "nonclaims": sorted(LEVEL2_NONCLAIMS),
        "processed_output": "benchmarks/processed/su2-level2-summary.json",
        "raw_samples_output": "benchmarks/processed/su2-level2-raw.json",
        "charts": [
            {"id": value, "output": f"benchmarks/plots/{value}.svg"}
            for value in (
                "latency",
                "throughput",
                "error_drift",
                "raw_paired_samples",
                "timing_breakdown",
                "stability",
            )
        ],
    }

    validate_manifest(release, repo_root=tmp_path)
    release_path = tmp_path / "benchmarks/manifests/wormhole-su2-compose-level2.json"
    release_path.write_text(json.dumps(release, indent=2) + "\n")
    first_output = tmp_path / "generated-first"
    second_output = tmp_path / "generated-second"
    outputs = generate_release(
        release_path,
        repo_root=tmp_path,
        destination_root=first_output,
    )
    assert outputs == generate_release(
        release_path,
        repo_root=tmp_path,
        destination_root=second_output,
    )
    assert all(
        (first_output / path).read_bytes() == (second_output / path).read_bytes()
        for path in outputs
    )

    release["artifacts"] = [
        artifact
        for artifact in release["artifacts"]
        if artifact["role"] != "stability-qualification"
    ]
    try:
        validate_manifest(release, repo_root=tmp_path)
    except ValueError as exc:
        assert "governance" in str(exc) or "qualification" in str(exc)
    else:
        raise AssertionError("Level 2 release accepted without hash-bound qualification")
