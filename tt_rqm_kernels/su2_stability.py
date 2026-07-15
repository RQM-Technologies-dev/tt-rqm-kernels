"""Deterministic stability qualification for SU2ComposeBench sessions."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

from tt_rqm_kernels.benchmark_integrity import IntegrityError
from tt_rqm_kernels.hardware_session import compare_device_health, validate_device_health


SCHEMA = "tt-rqm-su2-compose-stability-qualification.v1"
PREREGISTRATION_SCHEMA = "tt-rqm-su2-compose-stability-preregistration.v1"
PREREGISTRATION_SCHEMA_V2 = "tt-rqm-su2-compose-stability-preregistration.v2"
PREREGISTRATION_SCHEMA_V3 = "tt-rqm-su2-compose-stability-preregistration.v3"
SESSION_SCHEMAS = {
    "tt-rqm-su2-compose-session.v1",
    "tt-rqm-su2-compose-session.v2",
    "tt-rqm-su2-compose-session.v3",
}
DEFAULT_PREREGISTRATION = Path("benchmarks/manifests/su2-compose-stability-preregistration.json")
CASES = (
    (32768, 8),
    (8192, 32),
    (2048, 128),
    (512, 512),
    (1024, 128),
    (4096, 128),
    (16384, 128),
    (65536, 128),
)
REQUIRED_ROLES = {
    "candidate-identity",
    "environment",
    "exact-command",
    "hardware-report",
    "hardware-report-summary",
    "input-hashes",
    "post-device-health",
    "pre-device-health",
    "candidate-stderr",
    "candidate-stdout",
}
V3_REQUIRED_ROLES = REQUIRED_ROLES | {
    "host-state-post",
    "host-state-pre",
    "runtime-cache-inventory",
}
V3_HOST_IDENTITY_KEYS = (
    "cpu_model",
    "inherited_cpu_affinity",
    "requested_candidate_cpu_affinity",
    "process_nice",
    "cpu_governors",
    "numa_nodes",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_v3_pilot_repeat_counts(
    path: Path,
    *,
    repo_root: Path | None = None,
) -> dict[tuple[int, int], int]:
    root = (repo_root or Path.cwd()).resolve()
    payload = json.loads(_resolve(root, path).read_text(encoding="utf-8"))
    _require(
        payload.get("schema") == "tt-rqm-su2-compose-v3-pilot-repeat-counts.v1",
        "v3 pilot repeat-count schema mismatch",
    )
    _require(payload.get("benchmark_mode") == "fused_stability", "pilot plan must be fused-only")
    _require(payload.get("designated_stability_evidence") is False, "pilot plan cannot be designated")
    source = Path(str(payload.get("source_report")))
    source_path = _resolve(root, source)
    _require(source_path.is_file(), "pilot repeat-count source report is missing")
    _require(
        sha256_file(source_path) == payload.get("source_report_sha256"),
        "pilot repeat-count source report hash mismatch",
    )
    cases = payload.get("cases")
    _require(isinstance(cases, list), "pilot repeat counts are missing")
    _require(
        [(case.get("B"), case.get("K")) for case in cases] == list(CASES),
        "pilot repeat-count case order changed",
    )
    values = {(int(case["B"]), int(case["K"])): int(case["repeat_count"]) for case in cases}
    _require(all(value > 0 for value in values.values()), "pilot repeat counts must be positive")
    return values


def load_stability_preregistration(
    path: Path = DEFAULT_PREREGISTRATION,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    root = (repo_root or Path.cwd()).resolve()
    payload = json.loads(_resolve(root, path).read_text(encoding="utf-8"))
    return validate_stability_preregistration(payload, repo_root=root)


def validate_stability_preregistration(
    payload: Any,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    root = (repo_root or Path.cwd()).resolve()
    _require(isinstance(payload, dict), "SU2 stability preregistration must be an object")
    data = payload
    schema = data.get("schema")
    _require(
        schema in {PREREGISTRATION_SCHEMA, PREREGISTRATION_SCHEMA_V2, PREREGISTRATION_SCHEMA_V3},
        "SU2 stability schema mismatch",
    )
    if schema == PREREGISTRATION_SCHEMA_V3:
        return _validate_v3_preregistration(data, root=root)
    expected_status = (
        "frozen_before_designated_session_2"
        if schema == PREREGISTRATION_SCHEMA
        else "frozen_before_designated_session_1"
    )
    _require(data.get("status") == expected_status, "SU2 stability methodology is not frozen")
    session = data.get("session_contract")
    _require(
        session
        == {
            "all_designated_sessions_retained": True,
            "cold_start_host_process": True,
            "device_count": 1,
            "device_id": 0,
            "measured_pairs_per_case": 10,
            "no_discarded_performance_runs": True,
            "paired_order": "alternating_fused_first_unfused_first",
            "required_designated_sessions": 3,
            "warmup_pairs_per_case": 2,
        },
        "SU2 stability session contract changed",
    )
    statistic = data.get("statistic")
    _require(isinstance(statistic, dict), "SU2 stability statistic is missing")
    _require(
        statistic.get("required_metrics") == ["fused", "unfused", "ratio"],
        "both paths and paired ratio must qualify",
    )
    anchor_key = "first_session" if schema == PREREGISTRATION_SCHEMA else "calibration_experiment"
    first = data.get(anchor_key)
    _require(isinstance(first, dict), f"{anchor_key.replace('_', '-')} anchor is missing")
    first_path = _resolve(root, Path(str(first.get("report"))))
    _require(first_path.is_file(), "stability calibration report is missing")
    _require(
        sha256_file(first_path) == first.get("report_sha256"),
        "stability calibration report hash mismatch",
    )
    report = json.loads(first_path.read_text(encoding="utf-8"))
    if schema == PREREGISTRATION_SCHEMA_V2:
        _validate_v2_identity_and_inputs(data, report)
        profiler = data.get("profiler_decision")
        _require(isinstance(profiler, dict), "v2 profiler decision is missing")
        profiler_path = _resolve(root, Path(str(profiler.get("manifest"))))
        _require(profiler_path.is_file(), "v2 profiler evidence manifest is missing")
        _require(
            sha256_file(profiler_path) == profiler.get("manifest_sha256"),
            "v2 profiler evidence hash mismatch",
        )
        _require(
            profiler.get("architecture_decision") == "retain_exact_candidate",
            "v2 candidate was not retained by profiler decision",
        )
    cases = data.get("cases")
    _require(isinstance(cases, list), "SU2 stability case thresholds are missing")
    _require(
        [(case.get("B"), case.get("K")) for case in cases] == list(CASES),
        "SU2 stability case order changed",
    )
    for case, result in zip(cases, report.get("results", []), strict=True):
        limits = case.get("limits")
        _require(isinstance(limits, dict), "SU2 case limits are missing")
        samples = _metric_samples(result)
        for metric in ("fused", "unfused", "ratio"):
            expected = max(0.05, 2.0 * _dispersion(samples[metric]))
            _require(
                math.isclose(float(limits.get(metric, -1)), expected, rel_tol=0.0, abs_tol=1e-15),
                f"SU2 {case['B']}x{case['K']} {metric} limit is not derived from the calibration report",
            )
    invalid = data.get("invalid_session_rules")
    _require(
        isinstance(invalid, list) and len(invalid) == len(set(invalid)) >= 10,
        "invalid-session rules are incomplete",
    )
    return data


def _validate_v3_preregistration(data: dict[str, Any], *, root: Path) -> dict[str, Any]:
    status = data.get("status")
    _require(
        status in {"pilot_foundation_not_frozen", "frozen_before_designated_session_1"},
        "v3 status must be pilot-only or frozen before designated session 1",
    )
    _require(data.get("benchmark_mode") == "fused_stability", "v3 must isolate fused_stability")
    session = data.get("session_contract")
    _require(isinstance(session, dict), "v3 session contract is missing")
    expected_session = {
        "all_designated_sessions_retained": True,
        "cold_start_host_process": True,
        "device_count": 1,
        "device_id": 0,
        "isolated_empty_runtime_cache_per_session": True,
        "measured_samples_per_case": 10,
        "no_discarded_performance_runs": True,
        "one_synchronization_boundary_per_sample": True,
        "required_designated_sessions": 3,
        "separate_collector_invocations": True,
        "warmups_per_case": 5,
    }
    _require(session == expected_session, "v3 session contract changed")
    statistic = data.get("statistic")
    _require(isinstance(statistic, dict), "v3 stability statistic is missing")
    _require(statistic.get("required_metrics") == ["fused"], "v3 Level 2 requires fused only")
    _require(
        statistic.get("diagnostic_metrics") == ["unfused", "ratio"],
        "v3 diagnostics must name unfused and ratio",
    )
    _require(float(statistic.get("absolute_maximum_limit", -1)) == 0.10, "v3 maximum limit must be 10%")
    duration = data.get("raw_sample_duration_s")
    _require(
        duration == {"minimum": 0.025, "maximum": 0.05, "record_before_normalization": True},
        "v3 raw sample duration contract changed",
    )
    cases = data.get("cases")
    _require(isinstance(cases, list), "v3 cases are missing")
    _require(
        [(case.get("B"), case.get("K")) for case in cases] == list(CASES),
        "v3 case order changed",
    )
    for case in cases:
        limits = case.get("limits")
        _require(
            limits == {"fused": {"within_session_dispersion": 0.05, "cross_session_deviation": 0.05}},
            "v3 fused limits must remain 5%",
        )
        repeats = case.get("repeat_count")
        if status == "pilot_foundation_not_frozen":
            _require(repeats is None, "pilot-only v3 repeat counts must remain unset")
        else:
            _require(isinstance(repeats, int) and repeats > 0, "frozen v3 repeat counts must be positive")
    candidate = data.get("candidate")
    pilots = data.get("pilot_sessions")
    if status == "pilot_foundation_not_frozen":
        _require(candidate is None and pilots == [], "pilot-only v3 cannot freeze candidate or pilot evidence")
    else:
        _require(isinstance(candidate, dict), "frozen v3 candidate identity is missing")
        for key in (
            "sha256",
            "source_commit",
            "source_tree_sha256",
            "tt_metal_commit",
            "compiler_version",
            "runtime_version",
            "device",
            "pilot_collection_base_commit",
        ):
            _require(isinstance(candidate.get(key), str) and candidate[key], f"frozen v3 candidate {key} is missing")
        _require(
            pilots == ["pilot-1", "pilot-2", "pilot-3"],
            "frozen v3 requires the three retained pilot sessions",
        )
        frozen_at = data.get("frozen_at_utc")
        _require(
            isinstance(frozen_at, str) and frozen_at.endswith("+00:00"),
            "frozen v3 timestamp is missing or not UTC",
        )
        repeat_plan = load_v3_pilot_repeat_counts(Path(str(data.get("pilot_repeat_plan"))), repo_root=root)
        _require(
            [case["repeat_count"] for case in cases]
            == [repeat_plan[(batch, steps)] for batch, steps in CASES],
            "frozen v3 repeat counts differ from the retained pilot plan",
        )
        _validate_v3_host_contract(data)
        _validate_v3_pilot_evidence(data, root=root)
    invalid = data.get("invalid_session_rules")
    _require(isinstance(invalid, list) and len(invalid) == len(set(invalid)) >= 12, "v3 invalid-session rules are incomplete")
    return data


def _validate_v3_host_contract(data: Mapping[str, Any]) -> None:
    host = data.get("host_contract")
    _require(isinstance(host, dict), "frozen v3 host contract is missing")
    required = {
        "cpu_model",
        "inherited_cpu_affinity",
        "requested_candidate_cpu_affinity",
        "process_nice",
        "cpu_governors",
        "numa_nodes",
        "profiler_watcher_debug_disabled",
        "timing_environment",
    }
    _require(set(host) == required, "frozen v3 host contract changed")
    _require(host["requested_candidate_cpu_affinity"] == [24, 25, 26, 27], "v3 CPU affinity changed")
    _require(host["process_nice"] == 0, "v3 process nice changed")
    _require(host["profiler_watcher_debug_disabled"] is True, "v3 debug contract changed")
    _require(host["timing_environment"] == {}, "v3 timing environment changed")


def _validate_v3_pilot_evidence(data: Mapping[str, Any], *, root: Path) -> None:
    candidate = data["candidate"]
    assert isinstance(candidate, Mapping)
    evidence = data.get("pilot_evidence")
    _require(isinstance(evidence, dict), "frozen v3 pilot evidence is missing")
    assessment_path = _resolve(root, Path(str(evidence.get("assessment"))))
    _require(assessment_path.is_file(), "frozen v3 pilot assessment is missing")
    _require(
        sha256_file(assessment_path) == evidence.get("assessment_sha256"),
        "frozen v3 pilot assessment hash mismatch",
    )
    assessment = json.loads(assessment_path.read_text(encoding="utf-8"))
    _require(
        assessment.get("schema") == "tt-rqm-su2-compose-v3-pilot-assessment.v1",
        "frozen v3 pilot assessment schema mismatch",
    )
    _require(
        assessment.get("non_designated_pilot_only") is True
        and assessment.get("stable_benchmark") is False
        and assessment.get("qualification_passed") is False,
        "pilot assessment makes an invalid stability claim",
    )
    _require(
        assessment.get("ready_to_freeze_v3") is True
        and assessment.get("all_cases_within_preferred_5_percent") is True
        and assessment.get("rejected_gates") == [],
        "pilot assessment did not pass the frozen readiness gate",
    )
    pilots = evidence.get("sessions")
    _require(isinstance(pilots, list) and len(pilots) == 3, "frozen v3 pilot evidence is incomplete")
    ids = [str(pilot.get("id")) for pilot in pilots if isinstance(pilot, dict)]
    _require(ids == ["pilot-1", "pilot-2", "pilot-3"], "frozen v3 pilot IDs changed")
    _require(assessment.get("session_ids") == ids, "pilot assessment session IDs changed")
    for pilot in pilots:
        _require(isinstance(pilot, dict), "frozen v3 pilot entry is malformed")
        manifest_path = _resolve(root, Path(str(pilot.get("manifest"))))
        _require(manifest_path.is_file(), "frozen v3 pilot manifest is missing")
        _require(
            sha256_file(manifest_path) == pilot.get("manifest_sha256"),
            "frozen v3 pilot manifest hash mismatch",
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _require(manifest.get("session_id") == pilot.get("id"), "pilot manifest ID mismatch")
        _require(
            manifest.get("collection_status") == "passed"
            and manifest.get("designated_stability_session") is False
            and manifest.get("benchmark_mode") == "fused_stability",
            "pilot manifest contract mismatch",
        )
        _require(manifest.get("candidate_sha256") == candidate.get("sha256"), "pilot candidate hash mismatch")
        _require(
            manifest.get("tt_metal_commit") == candidate.get("tt_metal_commit"),
            "pilot TT-Metal commit mismatch",
        )
        _require(
            manifest.get("execution_source_commit") == candidate.get("pilot_collection_base_commit"),
            "pilot collection base commit mismatch",
        )
        _validate_imported_cache_inventory(manifest_path.parent)
    conformance = evidence.get("fused_conformance")
    _require(isinstance(conformance, dict), "frozen v3 conformance evidence is missing")
    manifest_path = _resolve(root, Path(str(conformance.get("manifest"))))
    _require(manifest_path.is_file(), "frozen v3 conformance manifest is missing")
    _require(
        sha256_file(manifest_path) == conformance.get("manifest_sha256"),
        "frozen v3 conformance manifest hash mismatch",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(
        manifest.get("collection_status") == "passed"
        and manifest.get("designated_stability_session") is False
        and manifest.get("benchmark_mode") == "fused_stability",
        "fused conformance manifest contract mismatch",
    )
    _require(manifest.get("candidate_sha256") == candidate.get("sha256"), "conformance candidate hash mismatch")
    report = json.loads((manifest_path.parent / "report.json").read_text(encoding="utf-8"))
    _require(
        report.get("benchmark_stage") == "conformance"
        and report.get("benchmark_mode") == "fused_stability"
        and all("unfused" not in result for result in report.get("results", [])),
        "fused conformance report contract mismatch",
    )
    remote = evidence.get("remote_assessment")
    _require(isinstance(remote, dict), "retained remote pilot assessment is missing")
    remote_path = _resolve(root, Path(str(remote.get("path"))))
    _require(remote_path.is_file(), "retained remote pilot assessment file is missing")
    _require(
        sha256_file(remote_path) == remote.get("sha256"),
        "retained remote pilot assessment hash mismatch",
    )


def _validate_imported_cache_inventory(session_dir: Path) -> None:
    cache_root = session_dir / "tt-metal-cache"
    inventory_path = session_dir / "cache-inventory.json"
    _require(cache_root.is_dir() and inventory_path.is_file(), "imported runtime cache evidence is missing")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    files = inventory.get("files")
    _require(isinstance(files, list) and len(files) == inventory.get("file_count"), "runtime cache inventory is malformed")
    for entry in files:
        _require(isinstance(entry, dict), "runtime cache inventory entry is malformed")
        path = (cache_root / str(entry.get("path"))).resolve()
        try:
            path.relative_to(cache_root.resolve())
        except ValueError as exc:
            raise IntegrityError("runtime cache inventory path escapes session") from exc
        _require(path.is_file(), "imported runtime cache file is missing")
        _require(path.stat().st_size == entry.get("size_bytes"), "imported runtime cache size mismatch")
        _require(sha256_file(path) == entry.get("sha256"), "imported runtime cache hash mismatch")


def _validate_v2_identity_and_inputs(data: Mapping[str, Any], report: Mapping[str, Any]) -> None:
    candidate = data.get("candidate")
    _require(isinstance(candidate, dict), "v2 stability candidate identity is missing")
    report_candidate = report.get("provenance", {}).get("candidate", {})
    expected_identity = {
        "sha256": report_candidate.get("candidate_sha256"),
        "source_commit": report_candidate.get("repository_commit"),
        "tt_metal_commit": report_candidate.get("tt_metal_commit"),
        "compiler_version": report_candidate.get("compiler_version"),
        "runtime_version": report_candidate.get("runtime_version"),
        "device": "tenstorrent/wormhole-device-0",
        "source_tree_sha256": "6fdb52dec6c2de283bdc7e7a21351903ab3e8bd694f6acf27e161724ee2aeea8",
    }
    _require(candidate == expected_identity, "v2 stability candidate identity mismatch")
    calibration = data.get("calibration_experiment")
    _require(
        isinstance(calibration, dict) and calibration.get("designated_stability_session") is False,
        "calibration experiment cannot be designated session 1",
    )
    expected_inputs = data.get("inputs")
    _require(isinstance(expected_inputs, list), "v2 deterministic input hashes are missing")
    _require(
        [(value.get("B"), value.get("K")) for value in expected_inputs] == list(CASES),
        "v2 deterministic input order mismatch",
    )
    _require(
        [value.get("case_id") for value in expected_inputs]
        == [result.get("case_id") for result in report.get("results", [])],
        "v2 calibration input identity mismatch",
    )
    for value in expected_inputs:
        for key in ("rotors_sha256", "phases_sha256"):
            observed = value.get(key)
            _require(
                isinstance(observed, str)
                and len(observed) == 64
                and all(character in "0123456789abcdef" for character in observed),
                f"invalid v2 input hash: {key}",
            )


def qualify_stability(
    manifest_paths: Sequence[Path],
    *,
    preregistration_path: Path = DEFAULT_PREREGISTRATION,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Analyze every designated session; no failed or noisy run is dropped."""

    root = (repo_root or Path.cwd()).resolve()
    normalized_manifest_paths = [
        str(_resolve(root, path).relative_to(root)) for path in manifest_paths
    ]
    preregistration = load_stability_preregistration(preregistration_path, repo_root=root)
    if (
        preregistration.get("schema") == PREREGISTRATION_SCHEMA_V3
        and preregistration.get("status") != "frozen_before_designated_session_1"
    ):
        raise IntegrityError("v3 pilot foundation is not frozen and cannot qualify designated sessions")
    analyses: list[dict[str, Any]] = []
    identities: list[tuple[str, ...] | None] = []
    input_identities: list[tuple[str, ...] | None] = []
    for path in manifest_paths:
        try:
            analysis, identity, input_identity = _analyze_session(path, root=root, require_designated=True)
        except Exception as exc:
            analysis = {
                "manifest": str(path),
                "session_id": path.parent.name,
                "performance_report": None,
                "passed_input_gates": False,
                "rejected_gates": [f"{type(exc).__name__}: {exc}"],
                "cases": [],
            }
            identity = None
            input_identity = None
        analyses.append(analysis)
        identities.append(identity)
        input_identities.append(input_identity)

    required = preregistration["session_contract"]["required_designated_sessions"]
    rejected: list[str] = []
    if len(manifest_paths) != required:
        rejected.append(f"exactly {required} designated cold-start sessions are required")
    ids = [str(analysis["session_id"]) for analysis in analyses]
    if len(ids) != len(set(ids)):
        rejected.append("session IDs are not distinct")
    reports = [analysis.get("performance_report") for analysis in analyses]
    if None in reports or len(reports) != len(set(reports)):
        rejected.append("designated sessions do not reference distinct performance reports")
    valid_identities = [value for value in identities if value is not None]
    if len(valid_identities) != len(analyses) or len(set(valid_identities)) != 1:
        rejected.append("candidate, source, TT-Metal, compiler, or runtime identity differs")
    valid_inputs = [value for value in input_identities if value is not None]
    comparable_inputs = (
        [tuple(item.split(":", 1)[0] for item in value) for value in valid_inputs]
        if preregistration.get("schema") == PREREGISTRATION_SCHEMA
        else valid_inputs
    )
    if len(comparable_inputs) != len(analyses) or len(set(comparable_inputs)) != 1:
        rejected.append("deterministic input identity differs")
    if preregistration.get("schema") == PREREGISTRATION_SCHEMA_V2:
        expected_identity = preregistration["candidate"]
        frozen_identity = (
            expected_identity["sha256"],
            expected_identity["source_commit"],
            expected_identity["tt_metal_commit"],
            expected_identity["compiler_version"],
            expected_identity["runtime_version"],
        )
        if not valid_identities or valid_identities[0] != frozen_identity:
            rejected.append("designated session identity differs from v2 preregistration")
        expected_inputs = tuple(
            f"{value['case_id']}:{value['rotors_sha256']}:{value['phases_sha256']}"
            for value in preregistration["inputs"]
        )
        if not valid_inputs or valid_inputs[0] != expected_inputs:
            rejected.append("designated session inputs differ from v2 preregistration")
    if preregistration.get("schema") == PREREGISTRATION_SCHEMA_V3:
        expected_candidate = preregistration["candidate"]
        expected_identity = (
            expected_candidate["sha256"],
            expected_candidate["source_commit"],
            expected_candidate["tt_metal_commit"],
            expected_candidate["compiler_version"],
            expected_candidate["runtime_version"],
        )
        if not valid_identities or valid_identities[0] != expected_identity:
            rejected.append("designated session identity differs from frozen v3 candidate")
        source_tree_hashes = [analysis.get("source_tree_sha256") for analysis in analyses]
        if (
            None in source_tree_hashes
            or len(set(source_tree_hashes)) != 1
            or source_tree_hashes[0] != expected_candidate["source_tree_sha256"]
        ):
            rejected.append("designated source tree differs from frozen v3 candidate")
        host_identities = [analysis.get("host_identity") for analysis in analyses]
        if None in host_identities or len(set(tuple(value) for value in host_identities if value)) != 1:
            rejected.append("frozen host parameters differ between designated sessions")
        host_contract = preregistration["host_contract"]
        expected_host_identity = tuple(
            json.dumps(host_contract[key], sort_keys=True)
            for key in V3_HOST_IDENTITY_KEYS
        )
        if not host_identities or host_identities[0] != expected_host_identity:
            rejected.append("designated sessions differ from the frozen host contract")
        if any(analysis.get("timing_environment") != host_contract["timing_environment"] for analysis in analyses):
            rejected.append("designated timing environment differs from frozen host contract")
        if any(analysis.get("profiler_watcher_debug_disabled") is not True for analysis in analyses):
            rejected.append("designated profiler, watcher, or debug contract changed")
        cache_paths = [analysis.get("runtime_cache_path") for analysis in analyses]
        if None in cache_paths or len(cache_paths) != len(set(cache_paths)):
            rejected.append("designated sessions did not use distinct runtime cache roots")

    qualified_cases: list[dict[str, Any]] = []
    required_metrics = tuple(preregistration["statistic"]["required_metrics"])
    for threshold in preregistration["cases"]:
        batch, steps = int(threshold["B"]), int(threshold["K"])
        case_rejected: list[str] = []
        metrics: dict[str, Any] = {}
        for metric in required_metrics:
            limit_spec = threshold["limits"][metric]
            within_limit = float(
                limit_spec["within_session_dispersion"] if isinstance(limit_spec, dict) else limit_spec
            )
            cross_limit = float(
                limit_spec["cross_session_deviation"] if isinstance(limit_spec, dict) else limit_spec
            )
            session_statistics = []
            for analysis in analyses:
                case = next(
                    (
                        value
                        for value in analysis["cases"]
                        if (value["B"], value["K"]) == (batch, steps)
                    ),
                    None,
                )
                if case is None:
                    case_rejected.append(f"{analysis['session_id']}: missing {batch}x{steps}")
                    continue
                value = case[metric]
                session_statistics.append(
                    {
                        "session_id": analysis["session_id"],
                        "median_s": value["median_s"],
                        "p95_s": value["p95_s"],
                        "within_session_dispersion": value["dispersion"],
                    }
                )
                if value["dispersion"] > within_limit:
                    case_rejected.append(
                        f"{analysis['session_id']}: {metric} within-session dispersion exceeds {within_limit:.9f}"
                    )
                if preregistration.get("schema") == PREREGISTRATION_SCHEMA_V3:
                    raw = case.get("raw_fused_samples_s", [])
                    minimum = float(preregistration["raw_sample_duration_s"]["minimum"])
                    maximum = float(preregistration["raw_sample_duration_s"]["maximum"])
                    if len(raw) != 10 or any(value < minimum or value > maximum for value in raw):
                        case_rejected.append(
                            f"{analysis['session_id']}: fused raw sample duration is outside {minimum:.3f}-{maximum:.3f}s"
                        )
            medians = [value["median_s"] for value in session_statistics]
            median_across = statistics.median(medians) if medians else None
            cross = []
            if median_across is not None:
                for value in session_statistics:
                    deviation = abs(value["median_s"] - median_across) / median_across
                    cross.append(
                        {
                            "session_id": value["session_id"],
                            "relative_deviation": deviation,
                        }
                    )
                    if deviation > cross_limit:
                        case_rejected.append(
                            f"{value['session_id']}: {metric} cross-session deviation exceeds {cross_limit:.9f}"
                        )
            metrics[metric] = {
                "within_session_limit": within_limit,
                "cross_session_limit": cross_limit,
                **({"limit": within_limit} if within_limit == cross_limit else {}),
                "session_statistics": session_statistics,
                "median_across_sessions_s": median_across,
                "cross_session_deviation": cross,
            }
        qualified_cases.append(
            {
                "B": batch,
                "K": steps,
                "metrics": metrics,
                "passed": len(analyses) == required and not case_rejected,
                "rejected_gates": case_rejected,
            }
        )

    for analysis in analyses:
        rejected.extend(
            f"{analysis['session_id']}: {reason}" for reason in analysis["rejected_gates"]
        )
    passed = (
        len(manifest_paths) == required
        and all(analysis["passed_input_gates"] for analysis in analyses)
        and all(case["passed"] for case in qualified_cases)
        and not rejected
    )
    return {
        "schema": SCHEMA,
        "benchmark_id": preregistration["benchmark_id"],
        "preregistration": str(preregistration_path),
        "required_designated_sessions": required,
        "session_ids": ids,
        "session_manifests": normalized_manifest_paths,
        "identity": None if not valid_identities else list(valid_identities[0]),
        "sessions": analyses,
        "cases": qualified_cases,
        "rejected_gates": rejected,
        "qualification_passed": passed,
        "stable_benchmark": passed,
    }


def write_qualification(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def assess_v3_pilots(
    manifest_paths: Sequence[Path],
    *,
    preregistration_path: Path,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Assess three non-designated fused-only pilots without making a stability claim."""

    root = (repo_root or Path.cwd()).resolve()
    preregistration = load_stability_preregistration(preregistration_path, repo_root=root)
    _require(
        preregistration.get("schema") == PREREGISTRATION_SCHEMA_V3,
        "pilot assessment requires the v3 preregistration",
    )
    analyses: list[dict[str, Any]] = []
    identities: list[tuple[str, ...]] = []
    input_identities: list[tuple[str, ...]] = []
    rejected: list[str] = []
    for path in manifest_paths:
        try:
            analysis, identity, input_identity = _analyze_session(
                path, root=root, require_designated=False
            )
        except Exception as exc:
            rejected.append(f"{path}: {type(exc).__name__}: {exc}")
            continue
        analyses.append(analysis)
        identities.append(identity)
        input_identities.append(input_identity)
        rejected.extend(f"{analysis['session_id']}: {reason}" for reason in analysis["rejected_gates"])
    if len(manifest_paths) != 3 or len(analyses) != 3:
        rejected.append("exactly three complete non-designated pilot sessions are required")
    ids = [analysis["session_id"] for analysis in analyses]
    if len(ids) != len(set(ids)):
        rejected.append("pilot session IDs are not distinct")
    if identities and len(set(identities)) != 1:
        rejected.append("pilot candidate or runtime identity differs")
    if input_identities and len(set(input_identities)) != 1:
        rejected.append("pilot deterministic input identity differs")
    host_identities = [analysis.get("host_identity") for analysis in analyses]
    if host_identities and (None in host_identities or len(set(host_identities)) != 1):
        rejected.append("pilot frozen host parameters differ")
    cache_paths = [analysis.get("runtime_cache_path") for analysis in analyses]
    if cache_paths and (None in cache_paths or len(cache_paths) != len(set(cache_paths))):
        rejected.append("pilot runtime cache roots are not distinct")

    minimum = float(preregistration["raw_sample_duration_s"]["minimum"])
    maximum = float(preregistration["raw_sample_duration_s"]["maximum"])
    cases: list[dict[str, Any]] = []
    for batch, steps in CASES:
        session_statistics = []
        case_rejected: list[str] = []
        for analysis in analyses:
            case = next(
                (value for value in analysis["cases"] if (value["B"], value["K"]) == (batch, steps)),
                None,
            )
            if case is None:
                case_rejected.append(f"{analysis['session_id']}: missing case")
                continue
            fused = case["fused"]
            raw = case["raw_fused_samples_s"]
            duration_ok = len(raw) == 10 and all(minimum <= value <= maximum for value in raw)
            if not duration_ok:
                case_rejected.append(
                    f"{analysis['session_id']}: raw sample duration outside {minimum:.3f}-{maximum:.3f}s"
                )
            session_statistics.append(
                {
                    "session_id": analysis["session_id"],
                    "median_s": fused["median_s"],
                    "p95_s": fused["p95_s"],
                    "within_session_dispersion": fused["dispersion"],
                    "raw_sample_duration_min_s": min(raw) if raw else None,
                    "raw_sample_duration_max_s": max(raw) if raw else None,
                }
            )
        medians = [value["median_s"] for value in session_statistics]
        across = statistics.median(medians) if medians else None
        deviations = []
        if across is not None:
            deviations = [
                {
                    "session_id": value["session_id"],
                    "relative_deviation": abs(value["median_s"] - across) / across,
                }
                for value in session_statistics
            ]
        observed = [value["within_session_dispersion"] for value in session_statistics] + [
            value["relative_deviation"] for value in deviations
        ]
        observed_max = max(observed) if observed else math.inf
        cases.append(
            {
                "B": batch,
                "K": steps,
                "session_statistics": session_statistics,
                "median_across_sessions_s": across,
                "cross_session_deviation": deviations,
                "observed_maximum": observed_max,
                "within_absolute_maximum_10_percent": observed_max <= 0.10 and not case_rejected,
                "within_preferred_5_percent": observed_max <= 0.05 and not case_rejected,
                "rejected_gates": case_rejected,
            }
        )
    ready = not rejected and all(case["within_absolute_maximum_10_percent"] for case in cases)
    return {
        "schema": "tt-rqm-su2-compose-v3-pilot-assessment.v1",
        "benchmark_id": preregistration["benchmark_id"],
        "preregistration": str(preregistration_path),
        "non_designated_pilot_only": True,
        "stable_benchmark": False,
        "qualification_passed": False,
        "session_ids": ids,
        "sessions": analyses,
        "cases": cases,
        "rejected_gates": rejected,
        "ready_to_freeze_v3": ready,
        "all_cases_within_preferred_5_percent": ready
        and all(case["within_preferred_5_percent"] for case in cases),
    }


def _analyze_session(
    manifest_path: Path,
    *,
    root: Path,
    require_designated: bool,
) -> tuple[dict[str, Any], tuple[str, ...], tuple[str, ...]]:
    manifest_path = _resolve(root, manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(manifest.get("schema") in SESSION_SCHEMAS, "unsupported SU2 session schema")
    modern = manifest.get("schema") in {"tt-rqm-su2-compose-session.v2", "tt-rqm-su2-compose-session.v3"}
    v3 = manifest.get("schema") == "tt-rqm-su2-compose-session.v3"
    host_identity: tuple[Any, ...] | None = None
    source_tree_sha256: str | None = None
    timing_environment: Any = None
    profiler_watcher_debug_disabled: Any = None
    rejected: list[str] = []
    if modern and manifest.get("collection_status") != "passed":
        rejected.append(f"designated collection failed: {manifest.get('failure')}")
    if manifest.get("cold_start_host_session") is not True:
        rejected.append("session is not a cold-start host process")
    if manifest.get("no_discarded_performance_runs") is not True:
        rejected.append("designated performance runs were discarded or not disclosed")
    individual_stable = (
        manifest.get("stable_benchmark")
        if modern
        else manifest.get("sample_contract", {}).get("stable_benchmark")
    )
    if individual_stable is not False:
        rejected.append("individual session must retain stable_benchmark=false")

    artifacts = manifest.get("artifacts")
    _require(isinstance(artifacts, list), "SU2 session artifact list is missing")
    roles: set[str] = set()
    role_paths: dict[str, Path] = {}
    report_path: Path | None = None
    for artifact in artifacts:
        _require(isinstance(artifact, dict), "SU2 session artifact is malformed")
        role = str(artifact.get("role"))
        roles.add(role)
        path = _artifact_path(root, manifest_path, str(artifact.get("path")))
        _require(path.is_file(), f"missing SU2 session artifact: {path}")
        _require(
            sha256_file(path) == artifact.get("sha256"),
            f"SU2 session artifact hash mismatch: {path}",
        )
        role_paths[role] = path
        if role in {"hardware-report", "raw-paired-performance-and-correctness"}:
            report_path = path
    required_roles = V3_REQUIRED_ROLES if v3 else REQUIRED_ROLES
    if modern and not required_roles.issubset(roles):
        rejected.append(f"session package is missing roles: {sorted(required_roles - roles)}")
    _require(report_path is not None, "SU2 session has no performance report")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    from tt_rqm_kernels.su2_benchmark_release import validate_su2_report

    validate_su2_report(report)
    lifecycle = report.get("lifecycle")
    if lifecycle != {"close_count": 1, "create_count": 1, "device_count": 1, "device_id": 0}:
        rejected.append("device lifecycle mismatch")
    if modern:
        if manifest.get("device_count") != 1 or manifest.get("device_id") != 0:
            rejected.append("session is not restricted to Wormhole device 0")
        if manifest.get("lifecycle") != lifecycle:
            rejected.append("session manifest lifecycle differs from report")
        if manifest.get("case_order") != [[b, k] for b, k in CASES]:
            rejected.append("session case order differs from preregistration")
        if require_designated and manifest.get("source_trees_clean") is not True:
            rejected.append("source trees were not recorded clean")
        if manifest.get("all_expected_paired_samples_retained") is not True:
            rejected.append("not every paired sample was retained")
        if v3:
            if manifest.get("benchmark_mode") != "fused_stability":
                rejected.append("v3 session did not isolate fused_stability mode")
            expected_designation = True if require_designated else False
            if manifest.get("designated_stability_session") is not expected_designation:
                rejected.append(
                    "v3 qualification received a non-designated pilot session"
                    if require_designated
                    else "v3 pilot assessment received a designated session"
                )
            if manifest.get("isolated_runtime_cache") is not True:
                rejected.append("v3 session did not use an isolated runtime cache")
        try:
            pre = role_paths["pre-device-health"].read_text(encoding="utf-8")
            post = role_paths["post-device-health"].read_text(encoding="utf-8")
            validate_device_health(pre, device_id=0)
            validate_device_health(post, device_id=0)
            compare_device_health(pre, post, device_id=0)
        except (KeyError, OSError, IntegrityError) as exc:
            rejected.append(f"device-health validation failed: {exc}")
        try:
            environment = json.loads(role_paths["environment"].read_text(encoding="utf-8"))
            source_tree_sha256 = environment.get("candidate", {}).get("source_tree_sha256")
            timing_environment = environment.get("timing_environment")
            profiler_watcher_debug_disabled = environment.get("profiler_watcher_debug_disabled")
            if require_designated and environment.get("repository", {}).get("status"):
                rejected.append("collector repository was dirty")
            if require_designated and environment.get("tt_metal", {}).get("status"):
                rejected.append("TT-Metal source tree was dirty")
            if v3 and environment.get("profiler_watcher_debug_disabled") is not True:
                rejected.append("profiler, watcher, or debug mode was not disabled")
        except (KeyError, OSError, json.JSONDecodeError) as exc:
            rejected.append(f"environment validation failed: {exc}")
        if v3:
            try:
                pre_host = json.loads(role_paths["host-state-pre"].read_text(encoding="utf-8"))
                post_host = json.loads(role_paths["host-state-post"].read_text(encoding="utf-8"))
                stable_keys = (
                    "cpu_model",
                    "inherited_cpu_affinity",
                    "requested_candidate_cpu_affinity",
                    "process_nice",
                    "cpu_governors",
                    "numa_nodes",
                )
                if any(pre_host.get(key) != post_host.get(key) for key in stable_keys):
                    rejected.append("frozen host state changed during the session")
                host_identity = tuple(json.dumps(pre_host.get(key), sort_keys=True) for key in stable_keys)
                cache = json.loads(role_paths["runtime-cache-inventory"].read_text(encoding="utf-8"))
                if cache.get("file_count", 0) <= 0:
                    rejected.append("runtime cache inventory is empty after collection")
                _validate_imported_cache_inventory(manifest_path.parent)
            except (KeyError, OSError, json.JSONDecodeError, TypeError) as exc:
                rejected.append(f"v3 host/cache validation failed: {exc}")
            except IntegrityError as exc:
                rejected.append(f"v3 host/cache validation failed: {exc}")

    candidate = report["provenance"]["candidate"]
    identity = (
        str(candidate["candidate_sha256"]),
        str(candidate["repository_commit"]),
        str(candidate["tt_metal_commit"]),
        str(candidate["compiler_version"]),
        str(candidate["runtime_version"]),
    )
    if modern:
        manifest_identity = (
            str(manifest.get("candidate_sha256")),
            str(manifest.get("execution_source_commit")),
            str(manifest.get("tt_metal_commit")),
        )
        if manifest_identity != identity[:3]:
            rejected.append("session manifest provenance differs from report")
    else:
        legacy = manifest.get("candidate", {})
        if (
            str(legacy.get("sha256")),
            str(legacy.get("repository_commit")),
            str(legacy.get("tt_metal_commit")),
        ) != identity[:3]:
            rejected.append("legacy session provenance differs from report")

    cases = []
    input_identity = tuple(
        (
            f"{result['case_id']}:{result['input_hashes']['rotors_sha256']}:"
            f"{result['input_hashes']['phases_sha256']}"
            if "input_hashes" in result
            else str(result["case_id"])
        )
        for result in report["results"]
    )
    if modern:
        try:
            recorded_inputs = json.loads(role_paths["input-hashes"].read_text(encoding="utf-8"))
            expected_inputs = [
                {
                    "B": result["B"],
                    "K": result["K"],
                    "case_id": result["case_id"],
                    **result["input_hashes"],
                }
                for result in report["results"]
            ]
            if recorded_inputs.get("seed") != 0 or recorded_inputs.get("cases") != expected_inputs:
                rejected.append("input-hash record differs from report")
        except (KeyError, OSError, json.JSONDecodeError, TypeError) as exc:
            rejected.append(f"input-hash validation failed: {exc}")
    for result in report["results"]:
        metric_samples = _metric_samples(result)
        case = {"B": int(result["B"]), "K": int(result["K"])}
        for metric, samples in metric_samples.items():
            if not all(math.isfinite(value) and value > 0 for value in samples):
                rejected.append(f"{result['B']}x{result['K']} {metric} timing is invalid")
            case[metric] = _summary(samples)
        case["raw_fused_samples_s"] = [
            float(value) for value in result["raw_candidate_timings_s"]["fused_samples"]
        ]
        cases.append(case)
    return (
        {
            "manifest": str(manifest_path.relative_to(root)),
            "session_id": str(manifest.get("session_id")),
            "performance_report": str(report_path.relative_to(root)),
            "legacy_session_manifest": not modern,
            "passed_input_gates": not rejected,
            "rejected_gates": rejected,
            "candidate_sha256": identity[0],
            "execution_source_commit": identity[1],
            "tt_metal_commit": identity[2],
            "compiler_version": identity[3],
            "runtime_version": identity[4],
            "input_case_ids": list(input_identity),
            "host_identity": host_identity,
            "source_tree_sha256": source_tree_sha256,
            "timing_environment": timing_environment,
            "profiler_watcher_debug_disabled": profiler_watcher_debug_disabled,
            "runtime_cache_path": manifest.get("runtime_cache_path") if v3 else None,
            "cases": cases,
        },
        identity,
        input_identity,
    )


def _metric_samples(result: Mapping[str, Any]) -> dict[str, list[float]]:
    fused = [float(value) for value in result["fused"]["timing_s"]["samples"]]
    if "unfused" not in result:
        _require(len(fused) == 10, "SU2 fused stability case must retain ten samples")
        return {"fused": fused}
    unfused = [float(value) for value in result["unfused"]["timing_s"]["samples"]]
    _require(len(fused) == len(unfused) == 10, "SU2 case must retain ten paired samples")
    return {
        "fused": fused,
        "unfused": unfused,
        "ratio": [a / b for a, b in zip(fused, unfused, strict=True)],
    }


def _summary(samples: Sequence[float]) -> dict[str, float | list[float]]:
    median = statistics.median(samples)
    p95 = sorted(samples)[math.ceil(0.95 * len(samples)) - 1]
    return {
        "median_s": median,
        "p95_s": p95,
        "dispersion": (p95 - median) / median,
        "samples": list(samples),
    }


def _dispersion(samples: Sequence[float]) -> float:
    return float(_summary(samples)["dispersion"])


def _artifact_path(root: Path, manifest_path: Path, value: str) -> Path:
    candidate = (manifest_path.parent / os.path.normpath(value)).resolve()
    if candidate.is_file():
        return candidate
    return _resolve(root, Path(value))


def _resolve(root: Path, path: Path) -> Path:
    candidate = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise IntegrityError(f"path escapes repository: {path}") from exc
    return candidate


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IntegrityError(message)
