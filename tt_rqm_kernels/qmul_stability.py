"""Deterministic three-session stability qualification for persistent qmul."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

from tt_rqm_kernels.benchmark_integrity import IntegrityError
from tt_rqm_kernels.backends.tenstorrent.qmul_persistent import validate_persistent_report
from tt_rqm_kernels.hardware_session import load_session_manifest, sha256_file


STABILITY_SCHEMA = "tt-rqm-benchmark-stability.v1"
ITEMS = (4096, 65536, 262144)
LIMITS = {4096: 0.104825, 65536: 0.05, 262144: 0.05}


def qualify_stability(manifest_paths: Sequence[Path]) -> dict[str, Any]:
    """Analyze all designated sessions without dropping failed or noisy runs."""

    analyses: list[dict[str, Any]] = []
    reports: list[Mapping[str, Any] | None] = []
    identities: list[tuple[str, str, str] | None] = []
    for path in manifest_paths:
        try:
            manifest = load_session_manifest(path)
            analysis, report, identity = _analyze_session(path, manifest)
        except Exception as exc:
            analysis = {
                "manifest": str(path),
                "session_id": path.parent.name,
                "passed_input_gates": False,
                "rejected_gates": [f"{type(exc).__name__}: {exc}"],
                "sizes": [],
            }
            report = None
            identity = None
        analyses.append(analysis)
        reports.append(report)
        identities.append(identity)

    global_rejections: list[str] = []
    ids = [str(value["session_id"]) for value in analyses]
    if len(manifest_paths) < 3:
        global_rejections.append("at least three designated cold-start sessions are required")
    if len(ids) != len(set(ids)):
        global_rejections.append("session IDs are not distinct")
    report_paths = [value.get("hardware_report") for value in analyses if value.get("hardware_report")]
    if len(report_paths) != len(set(report_paths)):
        global_rejections.append("designated sessions do not reference distinct hardware reports")
    valid_identities = [value for value in identities if value is not None]
    if len(set(valid_identities)) > 1:
        global_rejections.append("candidate, execution-source, or TT-Metal provenance differs")

    per_size: list[dict[str, Any]] = []
    for items in ITEMS:
        medians: list[float] = []
        p95_values: list[float] = []
        within: list[float] = []
        size_rejections: list[str] = []
        for analysis in analyses:
            match = next((value for value in analysis["sizes"] if value["items"] == items), None)
            if match is None:
                size_rejections.append(f"{analysis['session_id']}: missing N={items}")
                continue
            medians.append(match["median_s"])
            p95_values.append(match["p95_s"])
            within.append(match["within_session_dispersion"])
            if match["within_session_dispersion"] > LIMITS[items]:
                size_rejections.append(
                    f"{analysis['session_id']}: within-session dispersion exceeds {LIMITS[items]:.6f}"
                )
        median_across = statistics.median(medians) if medians else None
        cross_deviation = []
        if median_across is not None:
            for analysis, median_s in zip(
                [a for a in analyses if any(v["items"] == items for v in a["sizes"])],
                medians,
                strict=True,
            ):
                deviation = abs(median_s - median_across) / median_across
                cross_deviation.append({"session_id": analysis["session_id"], "relative_deviation": deviation})
                if deviation > LIMITS[items]:
                    size_rejections.append(
                        f"{analysis['session_id']}: cross-session deviation exceeds {LIMITS[items]:.6f}"
                    )
        per_size.append({
            "items": items,
            "limit": LIMITS[items],
            "session_medians_s": medians,
            "session_p95_s": p95_values,
            "within_session_dispersion": within,
            "median_across_sessions_s": median_across,
            "cross_session_deviation": cross_deviation,
            "passed": len(medians) >= 3 and not size_rejections,
            "rejected_gates": size_rejections,
        })

    for analysis in analyses:
        global_rejections.extend(
            f"{analysis['session_id']}: {reason}" for reason in analysis["rejected_gates"]
        )
    stable = (
        len(manifest_paths) >= 3
        and all(analysis["passed_input_gates"] for analysis in analyses)
        and all(value["passed"] for value in per_size)
        and not global_rejections
    )
    return {
        "schema": STABILITY_SCHEMA,
        "benchmark_id": "wormhole-qmul-fp32",
        "session_ids": ids,
        "session_manifests": [str(path) for path in manifest_paths],
        "required_cold_start_sessions": 3,
        "thresholds": {str(key): value for key, value in LIMITS.items()},
        "sessions": analyses,
        "sizes": per_size,
        "rejected_gates": global_rejections,
        "stable_benchmark": stable,
    }


def write_qualification(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _analyze_session(
    manifest_path: Path, manifest: Mapping[str, Any]
) -> tuple[dict[str, Any], Mapping[str, Any], tuple[str, str, str]]:
    rejected: list[str] = []
    if manifest.get("collection_status", "passed") != "passed":
        rejected.append(f"designated session collection failed: {manifest.get('failure')}")
    if manifest.get("cold_start_host_session") is not True:
        rejected.append("session is not marked as a cold-start host process")
    if manifest.get("stable_benchmark") is not False:
        rejected.append("individual session must retain stable_benchmark=false")
    if manifest.get("device_id") != 0 or manifest.get("device_count") != 1:
        rejected.append("stability qualification requires exactly Wormhole device 0")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise IntegrityError("session manifest has no artifact list")
    report_path: Path | None = None
    report_display_path: Path | None = None
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise IntegrityError("session artifact is malformed")
        display_path = Path(os.path.normpath(manifest_path.parent / str(artifact["path"])))
        path = display_path.resolve()
        if not path.is_file() or sha256_file(path) != artifact.get("sha256"):
            raise IntegrityError(f"session artifact hash mismatch: {path}")
        if artifact.get("role") == "hardware-report":
            report_path = path
            report_display_path = display_path
    if report_path is None:
        raise IntegrityError("session manifest has no hardware report")
    assert report_display_path is not None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    validate_persistent_report(report)
    if report.get("benchmark_stage") != "performance":
        rejected.append("session report is not a performance sweep")
    if report.get("case_items") != list(ITEMS) or report.get("seed") != 0:
        rejected.append("session case order or seed differs from preregistration")
    if report.get("lifecycle") != {"device_count": 1, "device_id": 0, "create_count": 1, "close_count": 1}:
        rejected.append("session lifecycle mismatch")

    sizes = []
    for result in report.get("results", []):
        items = int(result["items"])
        timing = result["timing"]["device_s"]
        median_s = float(timing["median"])
        p95_s = float(timing["p95"])
        samples = [float(value) for value in timing["samples"]]
        if result["iterations"] != 30 or result["warmup"] != 5 or len(samples) != 10:
            rejected.append(f"N={items} measurement contract mismatch")
        if not all(math.isfinite(value) and value > 0 for value in samples):
            rejected.append(f"N={items} contains invalid timing samples")
        correctness = result["correctness"]
        if correctness["failing_values"] or correctness["nonfinite_values"]:
            rejected.append(f"N={items} correctness failed")
        sizes.append({
            "items": items,
            "median_s": median_s,
            "p95_s": p95_s,
            "within_session_dispersion": (p95_s - median_s) / median_s,
            "samples_s": samples,
        })
    identity = (
        str(manifest.get("candidate_sha256")),
        str(manifest.get("execution_source_commit")),
        str(manifest.get("tt_metal_commit")),
    )
    return ({
        "manifest": str(manifest_path),
        "hardware_report": str(report_display_path),
        "session_id": str(manifest.get("session_id")),
        "passed_input_gates": not rejected,
        "rejected_gates": rejected,
        "candidate_sha256": identity[0],
        "execution_source_commit": identity[1],
        "tt_metal_commit": identity[2],
        "sizes": sizes,
    }, report, identity)
