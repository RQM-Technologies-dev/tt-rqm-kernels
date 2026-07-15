from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Callable

import pytest

from tt_rqm_kernels.benchmark_integrity import IntegrityError
from tt_rqm_kernels.su2_stability import (
    PREREGISTRATION_SCHEMA_V3,
    load_stability_preregistration,
    load_v3_pilot_repeat_counts,
    qualify_stability,
    sha256_file,
    validate_stability_preregistration,
    write_qualification,
)
from tt_rqm_kernels.su2_benchmark_release import (
    LEVEL2_NONCLAIMS,
    validate_manifest,
)
import tt_rqm_kernels.su2_stability as su2_stability


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


def test_v3_foundation_is_fused_only_and_frozen_before_collection() -> None:
    path = Path("benchmarks/manifests/su2-compose-stability-preregistration-v3.json")
    preregistration = load_stability_preregistration(path, repo_root=ROOT)
    assert preregistration["schema"] == PREREGISTRATION_SCHEMA_V3
    assert preregistration["status"] == "frozen_before_designated_session_1"
    assert preregistration["statistic"]["required_metrics"] == ["fused"]
    assert preregistration["statistic"]["diagnostic_metrics"] == ["unfused", "ratio"]
    assert [case["repeat_count"] for case in preregistration["cases"]] == [267, 90, 24, 7, 24, 24, 24, 12]
    assert preregistration["candidate"]["source_commit"] == "cd9118ccc342e7ba7143e34c0a2b570e82c1f4a6"
    assert preregistration["pilot_sessions"] == ["pilot-1", "pilot-2", "pilot-3"]
    qualification = qualify_stability([], preregistration_path=path, repo_root=ROOT)
    assert qualification["qualification_passed"] is False
    assert "exactly 3 designated cold-start sessions are required" in qualification["rejected_gates"]


def test_v3_frozen_evidence_hash_is_tamper_evident() -> None:
    path = ROOT / "benchmarks/manifests/su2-compose-stability-preregistration-v3.json"
    payload = json.loads(path.read_text())
    payload["pilot_evidence"]["assessment_sha256"] = "0" * 64
    with pytest.raises(IntegrityError, match="pilot assessment hash mismatch"):
        validate_stability_preregistration(payload, repo_root=ROOT)


def test_v3_qualification_requires_frozen_candidate_source_and_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = {
        "sha256": "candidate",
        "source_commit": "source",
        "source_tree_sha256": "tree",
        "tt_metal_commit": "metal",
        "compiler_version": "compiler",
        "runtime_version": "runtime",
    }
    host = {
        "cpu_model": "cpu",
        "inherited_cpu_affinity": [0],
        "requested_candidate_cpu_affinity": [24, 25, 26, 27],
        "process_nice": 0,
        "cpu_governors": ["schedutil"],
        "numa_nodes": ["node0"],
        "profiler_watcher_debug_disabled": True,
        "timing_environment": {},
    }
    preregistration = {
        "schema": PREREGISTRATION_SCHEMA_V3,
        "status": "frozen_before_designated_session_1",
        "benchmark_id": "test-v3",
        "candidate": candidate,
        "host_contract": host,
        "session_contract": {"required_designated_sessions": 3},
        "statistic": {"required_metrics": ["fused"]},
        "raw_sample_duration_s": {"minimum": 0.025, "maximum": 0.05},
        "cases": [
            {
                "B": batch,
                "K": steps,
                "limits": {"fused": {"within_session_dispersion": 0.05, "cross_session_deviation": 0.05}},
            }
            for batch, steps in su2_stability.CASES
        ],
    }
    host_identity = tuple(
        json.dumps(host[key], sort_keys=True) for key in su2_stability.V3_HOST_IDENTITY_KEYS
    )
    identity = ("candidate", "source", "metal", "compiler", "runtime")

    def fake_analyze(path: Path, *, root: Path, require_designated: bool):
        index = int(path.name[-1])
        analysis = {
            "session_id": f"session-{index}",
            "performance_report": f"report-{index}.json",
            "passed_input_gates": True,
            "rejected_gates": [],
            "host_identity": host_identity,
            "source_tree_sha256": "tree",
            "timing_environment": {},
            "profiler_watcher_debug_disabled": True,
            "runtime_cache_path": f"cache-{index}",
            "cases": [
                {
                    "B": batch,
                    "K": steps,
                    "fused": {"median_s": 0.001, "p95_s": 0.00101, "dispersion": 0.01},
                    "raw_fused_samples_s": [0.03] * 10,
                }
                for batch, steps in su2_stability.CASES
            ],
        }
        return analysis, identity, ("inputs",)

    monkeypatch.setattr(
        su2_stability, "load_stability_preregistration", lambda *args, **kwargs: preregistration
    )
    monkeypatch.setattr(su2_stability, "_analyze_session", fake_analyze)
    paths = [Path("session-1"), Path("session-2"), Path("session-3")]
    passed = qualify_stability(paths, preregistration_path=Path("v3.json"), repo_root=ROOT)
    assert passed["qualification_passed"] is True
    assert passed["stable_benchmark"] is True

    def wrong_source(path: Path, *, root: Path, require_designated: bool):
        analysis, observed_identity, inputs = fake_analyze(path, root=root, require_designated=require_designated)
        if path.name == "session-2":
            analysis["source_tree_sha256"] = "wrong"
        return analysis, observed_identity, inputs

    monkeypatch.setattr(su2_stability, "_analyze_session", wrong_source)
    rejected = qualify_stability(paths, preregistration_path=Path("v3.json"), repo_root=ROOT)
    assert rejected["qualification_passed"] is False
    assert "designated source tree differs from frozen v3 candidate" in rejected["rejected_gates"]


def test_v3_pilot_repeat_plan_is_disclosed_and_hash_bound() -> None:
    repeats = load_v3_pilot_repeat_counts(
        Path("benchmarks/manifests/su2-compose-v3-pilot-repeat-counts.json"),
        repo_root=ROOT,
    )
    assert repeats[(32768, 8)] == 267
    assert repeats[(512, 512)] == 7
    assert len(repeats) == 8
    audit = (ROOT / "reports/tt_hardware_su2_compose_v3_foundation_audit.md").read_text()
    assert "ready_to_freeze_v3=true" in audit
    assert "stable_benchmark=false" in audit
    assert "No `su2-v3-level2-session-*` collection was performed" in audit


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


def test_historical_paired_qualification_cannot_publish_as_v3_level_two(tmp_path: Path) -> None:
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

    with pytest.raises(ValueError, match="fused_stability"):
        validate_manifest(release, repo_root=tmp_path)
