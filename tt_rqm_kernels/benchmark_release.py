"""Evidence-backed benchmark release manifests and deterministic plots."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

from tt_rqm_kernels.backends.tenstorrent.qmul_persistent import (
    validate_persistent_report,
)

RELEASE_SCHEMA = "tt-rqm-benchmark-release.v1"
DEFAULT_MANIFEST = Path("benchmarks/manifests/wormhole-qmul-level2.json")
BASE_NONCLAIMS = {
    "no_hardware_bandwidth_claim",
    "no_cpu_comparison",
    "no_energy_claim",
    "no_application_speedup_claim",
    "no_dual_device_claim",
    "no_tenstorrent_endorsement",
    "no_acceleration_claim",
}
LEVEL_ONE_NONCLAIMS = BASE_NONCLAIMS | {"no_stability_claim"}
EXPECTED_CHARTS = {
    "throughput",
    "timing_breakdown",
    "raw_samples",
    "correctness",
}


class BenchmarkReleaseError(ValueError):
    """Raised when a benchmark release is not evidence-complete."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkReleaseError(f"cannot read benchmark manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BenchmarkReleaseError("benchmark manifest must be a JSON object")
    return value


def validate_release(
    manifest_path: Path = DEFAULT_MANIFEST,
    *,
    repo_root: Path | None = None,
    verify_generated: bool = True,
) -> dict[str, Any]:
    """Validate evidence, claims, hashes, and generated release files."""

    root = (repo_root or Path.cwd()).resolve()
    path = _resolve_repo_path(root, manifest_path)
    manifest = load_manifest(path)
    validate_manifest(manifest, repo_root=root)

    if verify_generated:
        with tempfile.TemporaryDirectory(prefix="tt-rqm-benchmark-release-") as temp:
            generated = generate_release(
                manifest_path=path,
                repo_root=root,
                destination_root=Path(temp),
            )
            for relative in generated:
                committed = _resolve_repo_path(root, relative)
                expected = Path(temp) / relative
                _require(committed.is_file(), f"missing generated release file: {relative}")
                _require(
                    committed.read_bytes() == expected.read_bytes(),
                    f"generated release file is stale: {relative}",
                )
    return manifest


def validate_manifest(
    manifest: Mapping[str, Any], *, repo_root: Path | None = None
) -> None:
    """Validate an in-memory release manifest against repository evidence."""

    root = (repo_root or Path.cwd()).resolve()
    _require(manifest.get("schema") == RELEASE_SCHEMA, "benchmark release schema mismatch")
    _require_nonempty(manifest, "benchmark_id")
    _require_nonempty(manifest, "title")

    artifacts = manifest.get("artifacts")
    _require(isinstance(artifacts, list) and artifacts, "manifest requires artifacts")
    artifact_by_path: dict[str, Mapping[str, Any]] = {}
    for artifact in artifacts:
        _require(isinstance(artifact, dict), "each artifact must be an object")
        relative = _require_nonempty(artifact, "path")
        expected_hash = _require_nonempty(artifact, "sha256")
        _require(re.fullmatch(r"[0-9a-f]{64}", expected_hash) is not None, f"invalid SHA-256 for {relative}")
        artifact_path = _resolve_repo_path(root, Path(relative))
        _require(artifact_path.is_file(), f"missing benchmark artifact: {relative}")
        observed_hash = sha256_file(artifact_path)
        _require(observed_hash == expected_hash, f"artifact SHA-256 mismatch: {relative}")
        _require(relative not in artifact_by_path, f"duplicate artifact path: {relative}")
        artifact_by_path[relative] = artifact

    source_path = _require_nonempty(manifest, "primary_report")
    _require(source_path in artifact_by_path, "primary report is not a hashed artifact")
    report_path = _resolve_repo_path(root, Path(source_path))
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        validate_persistent_report(report)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise BenchmarkReleaseError(f"primary persistent report is invalid: {exc}") from exc
    _validate_report_contract(report)
    _validate_provenance(manifest, report)
    _validate_claim(manifest, report, artifact_by_path, root)
    _validate_nonclaims(manifest, artifacts)
    _validate_outputs(manifest)


def generate_release(
    manifest_path: Path = DEFAULT_MANIFEST,
    *,
    repo_root: Path | None = None,
    destination_root: Path | None = None,
) -> tuple[Path, ...]:
    """Generate the normalized summary and the four evidence-supported SVGs."""

    root = (repo_root or Path.cwd()).resolve()
    path = _resolve_repo_path(root, manifest_path)
    manifest = load_manifest(path)
    report_path = _resolve_repo_path(root, Path(str(manifest["primary_report"])))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    output_root = (destination_root or root).resolve()

    summary_path = Path(str(manifest["processed_output"]))
    summary_target = output_root / summary_path
    summary_target.parent.mkdir(parents=True, exist_ok=True)
    summary_target.write_text(
        json.dumps(build_processed_summary(manifest, report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    charts = {chart["id"]: Path(chart["output"]) for chart in manifest["charts"]}
    for output in charts.values():
        (output_root / output).parent.mkdir(parents=True, exist_ok=True)
    _render_plots(report, charts, output_root)
    return (summary_path, *(charts[chart_id] for chart_id in sorted(charts)))


def build_processed_summary(
    manifest: Mapping[str, Any], report: Mapping[str, Any]
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for result in report["results"]:
        cases.append(
            {
                "items": result["items"],
                "iterations": result["iterations"],
                "core_count": result["candidate_metadata"]["core_count"],
                "device_median_s": result["timing"]["device_s"]["median"],
                "device_p95_s": result["timing"]["device_s"]["p95"],
                "device_samples_s": result["timing"]["device_s"]["samples"],
                "throughput_qmul_per_s": result["throughput"],
                "logical_traffic_gb_per_s": result["effective_gb_per_s"],
                "estimated_flops_per_s": result["estimated_flops_per_s"],
                "validated_values": result["correctness"]["validated_values"],
                "failing_values": result["correctness"]["failing_values"],
                "nonfinite_values": result["correctness"]["nonfinite_values"],
                "whole_output_max_abs_error": result["correctness"]["whole_output_max_abs_error"],
                "atol": result["correctness"]["atol"],
                "rtol": result["correctness"]["rtol"],
                "phases_s": result["timing"]["phases_s"],
            }
        )
    return {
        "schema": "tt-rqm-benchmark-processed.v1",
        "benchmark_id": manifest["benchmark_id"],
        "claim": manifest["claim"],
        "source_report": manifest["primary_report"],
        "session_timing_s": report["session_timing"],
        "cases": cases,
    }


def _validate_report_contract(report: Mapping[str, Any]) -> None:
    _require(report.get("benchmark_stage") == "performance", "primary report must be a performance report")
    _require(report.get("stable_benchmark") is False, "current primary report must remain non-stable")
    _require(report.get("case_items") == [4096, 65536, 262144], "primary report case sweep mismatch")
    _require(report.get("repetitions") == 10, "primary report must contain ten samples per case")
    _require(report.get("lifecycle") == {"close_count": 1, "create_count": 1, "device_count": 1, "device_id": 0}, "primary report must use one device-0 lifecycle")
    for result in report.get("results", []):
        items = int(result["items"])
        iterations = int(result["iterations"])
        _require(result.get("performance_eligible") is True, "primary result must be architecture-eligible")
        _require(result.get("stable_benchmark") is False, "primary result must remain non-stable")
        _require(result["correctness"]["passed"] is True, "primary result failed correctness")
        _require(result["correctness"]["failing_values"] == 0, "primary result contains failing values")
        _require(result["correctness"]["nonfinite_values"] == 0, "primary result contains non-finite values")
        _require(result["estimated_flops"] == items * iterations * 28, "operation-count model mismatch")
        _require(result["estimated_total_bytes"] == items * iterations * 48, "logical-byte model mismatch")
        logical_rate = result["estimated_total_bytes"] / result["elapsed_s"] / 1e9
        _require(math.isclose(logical_rate, result["effective_gb_per_s"], rel_tol=1e-12), "logical traffic rate mismatch")


def _validate_provenance(manifest: Mapping[str, Any], report: Mapping[str, Any]) -> None:
    provenance = manifest.get("provenance")
    _require(isinstance(provenance, dict), "manifest requires provenance")
    observed = report["provenance"]["candidate"]
    for key in ("candidate_sha256", "repository_commit", "tt_metal_commit"):
        _require(provenance.get(key) == observed.get(key), f"manifest provenance mismatch: {key}")


def _validate_claim(
    manifest: Mapping[str, Any],
    report: Mapping[str, Any],
    artifact_by_path: Mapping[str, Mapping[str, Any]],
    root: Path,
) -> None:
    claim = manifest.get("claim")
    _require(isinstance(claim, dict), "manifest requires claim object")
    level = claim.get("level")
    _require(isinstance(level, int) and 0 <= level <= 5, "claim level must be 0 through 5")
    sessions = manifest.get("sessions")
    _require(isinstance(sessions, list) and sessions, "manifest requires public sessions")
    ids = [session.get("id") for session in sessions if isinstance(session, dict)]
    _require(len(ids) == len(sessions) and len(ids) == len(set(ids)), "session IDs must be unique strings")
    _require(all(isinstance(value, str) and value for value in ids), "session IDs must be unique strings")
    _require(claim.get("public_session_count") == len(sessions), "claim session count mismatch")
    session_paths: list[str] = []
    for session in sessions:
        report_path = session.get("performance_report")
        _require(report_path in artifact_by_path, f"session report is not a hashed artifact: {report_path}")
        session_paths.append(str(report_path))
        try:
            session_report = json.loads(
                _resolve_repo_path(root, Path(str(report_path))).read_text(encoding="utf-8")
            )
            validate_persistent_report(session_report)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise BenchmarkReleaseError(
                f"persistent session report is invalid: {report_path}: {exc}"
            ) from exc
        _require(
            session_report.get("benchmark_stage") == "performance",
            f"session report is not a performance report: {report_path}",
        )
        _require(
            session_report["provenance"]["candidate"]["candidate_sha256"]
            == manifest["provenance"]["candidate_sha256"],
            f"session candidate differs from release candidate: {report_path}",
        )
        _require(
            session_report["provenance"]["candidate"]["tt_metal_commit"]
            == manifest["provenance"]["tt_metal_commit"],
            f"session TT-Metalium commit differs from release pin: {report_path}",
        )
    _require(
        len(session_paths) == len(set(session_paths)),
        "independent sessions must reference distinct performance reports",
    )
    if level == 1:
        _require(len(sessions) == 1, "Claim Level 1 requires exactly one public session")
        _require(claim.get("stable_benchmark") is False, "Claim Level 1 must remain non-stable")
        _require(report.get("stable_benchmark") is False, "Claim Level 1 source must remain non-stable")
    if level >= 2:
        _require(len(sessions) >= 3, "Claim Level 2 requires at least three independent sessions")
        _require(claim.get("stable_benchmark") is True, "Claim Level 2 requires stable_benchmark=true")
        _require(all(session.get("qualification_passed") is True for session in sessions), "every Level 2 session must pass qualification")
        session_manifest_paths: list[str] = []
        for session in sessions:
            session_manifest_path = session.get("session_manifest")
            _require(
                session_manifest_path in artifact_by_path,
                f"session manifest is not a hashed artifact: {session_manifest_path}",
            )
            session_manifest_paths.append(str(session_manifest_path))
            _validate_session_manifest(
                session,
                manifest,
                root,
                str(session_manifest_path),
            )
        qualification = next(
            (
                artifact
                for artifact in artifact_by_path.values()
                if artifact.get("role") == "stability-qualification"
            ),
            None,
        )
        _require(qualification is not None, "Claim Level 2 requires a hashed stability qualification")
        try:
            qualification_payload = json.loads(
                _resolve_repo_path(root, Path(str(qualification["path"]))).read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise BenchmarkReleaseError(f"stability qualification is invalid: {exc}") from exc
        _require(
            qualification_payload.get("schema") == "tt-rqm-benchmark-stability.v1",
            "stability qualification schema mismatch",
        )
        _require(
            qualification_payload.get("stable_benchmark") is True,
            "stability qualification must set stable_benchmark=true",
        )
        _require(
            qualification_payload.get("session_ids") == ids,
            "stability qualification session IDs must match the release",
        )
        _require(
            qualification_payload.get("session_manifests") == session_manifest_paths,
            "stability qualification session manifests must match the release",
        )


def _validate_nonclaims(manifest: Mapping[str, Any], artifacts: list[Any]) -> None:
    nonclaims = manifest.get("nonclaims")
    _require(isinstance(nonclaims, list), "manifest requires nonclaims")
    level = manifest.get("claim", {}).get("level")
    expected = LEVEL_ONE_NONCLAIMS if level == 1 else BASE_NONCLAIMS
    _require(set(nonclaims) == expected, "manifest nonclaims are incomplete")
    has_ceiling = any(
        isinstance(artifact, dict) and artifact.get("role") == "measured-hardware-ceiling"
        for artifact in artifacts
    )
    if not has_ceiling:
        for value in _walk_values(manifest):
            normalized = str(value).lower().replace("-", "_")
            if "measured bandwidth" in normalized or "measured_bandwidth" in normalized:
                raise BenchmarkReleaseError(
                    "measured-bandwidth language requires a hashed hardware-ceiling artifact"
                )


def _validate_session_manifest(
    session: Mapping[str, Any],
    release: Mapping[str, Any],
    root: Path,
    relative: str,
) -> None:
    path = _resolve_repo_path(root, Path(relative))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkReleaseError(f"session manifest is invalid: {relative}: {exc}") from exc
    _require(
        payload.get("schema") in {"tt-rqm-benchmark-session.v1", "tt-rqm-benchmark-session.v2"},
        f"session manifest schema mismatch: {relative}",
    )
    _require(payload.get("session_id") == session.get("id"), f"session ID mismatch: {relative}")
    _require(payload.get("collection_status") == "passed", f"session did not pass: {relative}")
    _require(payload.get("cold_start_host_session") is True, f"session is not cold-start: {relative}")
    _require(payload.get("stable_benchmark") is False, f"individual session must remain non-stable: {relative}")
    _require(payload.get("device_id") == 0 and payload.get("device_count") == 1, f"session device scope mismatch: {relative}")
    _require(payload.get("candidate_sha256") == release["provenance"]["candidate_sha256"], f"session candidate mismatch: {relative}")
    _require(payload.get("tt_metal_commit") == release["provenance"]["tt_metal_commit"], f"session TT-Metalium mismatch: {relative}")

    hardware_report: Path | None = None
    artifacts = payload.get("artifacts")
    _require(isinstance(artifacts, list) and artifacts, f"session has no artifacts: {relative}")
    for artifact in artifacts:
        _require(isinstance(artifact, dict), f"malformed session artifact: {relative}")
        artifact_path = (path.parent / str(artifact.get("path"))).resolve()
        _require(artifact_path.is_file(), f"missing session artifact: {artifact_path}")
        _require(
            sha256_file(artifact_path) == artifact.get("sha256"),
            f"session artifact SHA-256 mismatch: {artifact_path}",
        )
        if artifact.get("role") == "hardware-report":
            hardware_report = artifact_path
    _require(hardware_report is not None, f"session has no hardware report: {relative}")
    expected_report = _resolve_repo_path(root, Path(str(session.get("performance_report"))))
    _require(hardware_report == expected_report, f"session report path mismatch: {relative}")


def _validate_outputs(manifest: Mapping[str, Any]) -> None:
    _require_nonempty(manifest, "processed_output")
    charts = manifest.get("charts")
    _require(isinstance(charts, list), "manifest requires charts")
    chart_ids = {chart.get("id") for chart in charts if isinstance(chart, dict)}
    _require(chart_ids == EXPECTED_CHARTS, "manifest must define exactly the four supported charts")
    for chart in charts:
        _require_nonempty(chart, "output")
        _require(str(chart["output"]).endswith(".svg"), "benchmark charts must be SVG")


def _render_plots(
    report: Mapping[str, Any], charts: Mapping[str, Path], output_root: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "svg.fonttype": "path",
            "svg.hashsalt": "tt-rqm-wormhole-qmul-v1",
        }
    )
    import matplotlib.pyplot as plt

    results = report["results"]
    sizes = [result["items"] for result in results]
    labels = [f"{size:,}" for size in sizes]
    color = "#4c78a8"
    accent = "#f58518"
    metadata = {
        "Date": None,
        "Creator": "tt-rqm-kernels deterministic benchmark generator",
    }

    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    throughput = [result["throughput"] / 1e6 for result in results]
    ax.plot(labels, throughput, marker="o", linewidth=2, color=color)
    ax.set_title("Persistent qmul throughput — one public session")
    ax.set_xlabel("Quaternion rows (N)")
    ax.set_ylabel("Throughput (million qmul/s)")
    ax.grid(axis="y", alpha=0.25)
    _save_svg(fig, output_root / charts["throughput"], metadata)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), constrained_layout=True)
    session = report["session_timing"]
    session_names = ["device create", "candidate session", "device close", "host end-to-end"]
    session_values = [
        session["device_create"],
        session["candidate_session"],
        session["device_close"],
        session["host_process_end_to_end_s"],
    ]
    axes[0].barh(session_names, session_values, color=[color, accent, color, "#54a24b"])
    axes[0].set_title("Session scopes (not additive)")
    axes[0].set_xlabel("Seconds")
    setup_values = []
    device_values = []
    readback_values = []
    for result in results:
        phases = result["timing"]["phases_s"]
        setup_values.append(
            phases["buffer_allocation"]
            + phases["program_build"]
            + phases["h2d"]
            + phases["prewarm_sync"]
            + phases["warmup"]
        )
        device_values.append(result["timing"]["device_s"]["median"])
        readback_values.append(phases["d2h"] + phases["cleanup"])
    x = range(len(labels))
    axes[1].bar(x, setup_values, label="setup + warmup", color=color)
    axes[1].bar(x, device_values, bottom=setup_values, label="median device sample", color=accent)
    bottoms = [a + b for a, b in zip(setup_values, device_values, strict=True)]
    axes[1].bar(x, readback_values, bottom=bottoms, label="readback + cleanup", color="#54a24b")
    axes[1].set_xticks(list(x), labels)
    axes[1].set_title("Recorded case phases")
    axes[1].set_xlabel("Quaternion rows (N)")
    axes[1].set_ylabel("Seconds")
    axes[1].legend(fontsize=8)
    _save_svg(fig, output_root / charts["timing_breakdown"], metadata)

    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    samples_ms = [
        [sample * 1000.0 for sample in result["timing"]["device_s"]["samples"]]
        for result in results
    ]
    ax.boxplot(samples_ms, tick_labels=labels, showmeans=True)
    for index, values in enumerate(samples_ms, start=1):
        offsets = [(point - 4.5) * 0.012 for point in range(len(values))]
        ax.scatter([index + offset for offset in offsets], values, s=18, color=color, zorder=3)
    ax.set_title("All ten device samples per size — one session, not stability")
    ax.set_xlabel("Quaternion rows (N)")
    ax.set_ylabel("Prepared-workload device time (ms)")
    ax.grid(axis="y", alpha=0.25)
    _save_svg(fig, output_root / charts["raw_samples"], metadata)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), constrained_layout=True)
    errors = [result["correctness"]["whole_output_max_abs_error"] for result in results]
    tolerance = [result["correctness"]["atol"] for result in results]
    axes[0].semilogy(labels, errors, marker="o", linewidth=2, color=color, label="max absolute error")
    axes[0].semilogy(labels, tolerance, linestyle="--", color=accent, label="absolute tolerance")
    axes[0].set_title("Whole-output error")
    axes[0].set_xlabel("Quaternion rows (N)")
    axes[0].set_ylabel("Absolute value")
    axes[0].legend(fontsize=8)
    validated = [result["correctness"]["validated_values"] for result in results]
    axes[1].bar(labels, validated, color=color)
    axes[1].set_title("Validated output values")
    axes[1].set_xlabel("Quaternion rows (N)")
    axes[1].set_ylabel("Values")
    axes[1].ticklabel_format(axis="y", style="plain")
    _save_svg(fig, output_root / charts["correctness"], metadata)


def _save_svg(fig: Any, path: Path, metadata: Mapping[str, Any]) -> None:
    import matplotlib.pyplot as plt

    fig.savefig(path, format="svg", metadata=metadata)
    plt.close(fig)
    raw = path.read_text(encoding="utf-8")
    generated_ids = list(
        dict.fromkeys(re.findall(r'id="([mp][0-9a-f]{10})"', raw))
    )
    for index, generated_id in enumerate(generated_ids):
        canonical_id = f"{generated_id[0]}tt_rqm_{index:03d}"
        raw = raw.replace(f'id="{generated_id}"', f'id="{canonical_id}"')
        raw = raw.replace(f"#{generated_id}", f"#{canonical_id}")
    normalized = "\n".join(line.rstrip() for line in raw.splitlines())
    path.write_text(normalized + "\n", encoding="utf-8")


def _resolve_repo_path(root: Path, path: Path) -> Path:
    if path.is_absolute():
        candidate = path.resolve()
    else:
        candidate = (root / path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise BenchmarkReleaseError(f"path escapes repository root: {path}") from exc
    return candidate


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BenchmarkReleaseError(message)


def _require_nonempty(value: Mapping[str, Any], key: str) -> str:
    observed = value.get(key)
    _require(isinstance(observed, str) and bool(observed.strip()), f"manifest requires non-empty {key}")
    return str(observed)


def _walk_values(value: Any):
    if isinstance(value, Mapping):
        for key, nested in value.items():
            yield key
            yield from _walk_values(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_values(nested)
    elif isinstance(value, str):
        yield value
