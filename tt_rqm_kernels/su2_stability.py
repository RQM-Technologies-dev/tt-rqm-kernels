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
SESSION_SCHEMAS = {
    "tt-rqm-su2-compose-session.v1",
    "tt-rqm-su2-compose-session.v2",
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


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        schema in {PREREGISTRATION_SCHEMA, PREREGISTRATION_SCHEMA_V2},
        "SU2 stability schema mismatch",
    )
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
    analyses: list[dict[str, Any]] = []
    identities: list[tuple[str, ...] | None] = []
    input_identities: list[tuple[str, ...] | None] = []
    for path in manifest_paths:
        try:
            analysis, identity, input_identity = _analyze_session(path, root=root)
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

    qualified_cases: list[dict[str, Any]] = []
    for threshold in preregistration["cases"]:
        batch, steps = int(threshold["B"]), int(threshold["K"])
        case_rejected: list[str] = []
        metrics: dict[str, Any] = {}
        for metric in ("fused", "unfused", "ratio"):
            limit = float(threshold["limits"][metric])
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
                if value["dispersion"] > limit:
                    case_rejected.append(
                        f"{analysis['session_id']}: {metric} within-session dispersion exceeds {limit:.9f}"
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
                    if deviation > limit:
                        case_rejected.append(
                            f"{value['session_id']}: {metric} cross-session deviation exceeds {limit:.9f}"
                        )
            metrics[metric] = {
                "limit": limit,
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


def _analyze_session(
    manifest_path: Path,
    *,
    root: Path,
) -> tuple[dict[str, Any], tuple[str, ...], tuple[str, ...]]:
    manifest_path = _resolve(root, manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(manifest.get("schema") in SESSION_SCHEMAS, "unsupported SU2 session schema")
    modern = manifest.get("schema") == "tt-rqm-su2-compose-session.v2"
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
    if modern and not REQUIRED_ROLES.issubset(roles):
        rejected.append(f"session package is missing roles: {sorted(REQUIRED_ROLES - roles)}")
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
        if manifest.get("source_trees_clean") is not True:
            rejected.append("source trees were not recorded clean")
        if manifest.get("all_expected_paired_samples_retained") is not True:
            rejected.append("not every paired sample was retained")
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
            if environment.get("repository", {}).get("status"):
                rejected.append("collector repository was dirty")
            if environment.get("tt_metal", {}).get("status"):
                rejected.append("TT-Metal source tree was dirty")
        except (KeyError, OSError, json.JSONDecodeError) as exc:
            rejected.append(f"environment validation failed: {exc}")

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
            "cases": cases,
        },
        identity,
        input_identity,
    )


def _metric_samples(result: Mapping[str, Any]) -> dict[str, list[float]]:
    fused = [float(value) for value in result["fused"]["timing_s"]["samples"]]
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
