"""Hash-bound SU2ComposeBench releases and deterministic evidence plots."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Mapping

from tt_rqm_kernels.benchmark_release import _save_svg


SCHEMA = "tt-rqm-su2-compose-release.v1"
DEFAULT_MANIFEST = Path("benchmarks/manifests/wormhole-su2-compose.json")
CASES = ((32768, 8), (8192, 32), (2048, 128), (512, 512),
         (1024, 128), (4096, 128), (16384, 128), (65536, 128))
CHART_IDS = {"latency", "throughput", "error_drift", "raw_paired_samples", "timing_breakdown"}
NONCLAIMS = {
    "no_stability_claim", "no_acceleration_claim", "no_cpu_comparison",
    "no_measured_bandwidth_claim", "no_energy_claim", "no_dual_device_claim",
    "no_full_device_side_hamiltonian_lowering_claim", "no_tenstorrent_endorsement",
}


class SU2ReleaseError(ValueError):
    """Raised when SU2ComposeBench release evidence is incomplete."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SU2ReleaseError(f"cannot read SU2 release manifest {path}: {exc}") from exc
    _require(isinstance(value, dict), "SU2 release manifest must be an object")
    return value


def validate_release(
    manifest_path: Path = DEFAULT_MANIFEST,
    *,
    repo_root: Path | None = None,
    verify_generated: bool = True,
) -> dict[str, Any]:
    root = (repo_root or Path.cwd()).resolve()
    path = _resolve(root, manifest_path)
    manifest = load_manifest(path)
    validate_manifest(manifest, repo_root=root)
    if verify_generated:
        with tempfile.TemporaryDirectory(prefix="tt-rqm-su2-release-") as temp:
            outputs = generate_release(path, repo_root=root, destination_root=Path(temp))
            for relative in outputs:
                committed = _resolve(root, relative)
                expected = Path(temp) / relative
                _require(committed.is_file(), f"missing generated SU2 file: {relative}")
                _require(committed.read_bytes() == expected.read_bytes(), f"stale generated SU2 file: {relative}")
    return manifest


def validate_manifest(manifest: Mapping[str, Any], *, repo_root: Path | None = None) -> None:
    root = (repo_root or Path.cwd()).resolve()
    _require(manifest.get("schema") == SCHEMA, "SU2 release schema mismatch")
    artifacts = manifest.get("artifacts")
    _require(isinstance(artifacts, list) and artifacts, "SU2 release requires artifacts")
    by_path: dict[str, Mapping[str, Any]] = {}
    for artifact in artifacts:
        _require(isinstance(artifact, dict), "SU2 artifact must be an object")
        relative = _text(artifact, "path")
        expected = _text(artifact, "sha256")
        path = _resolve(root, Path(relative))
        _require(path.is_file(), f"missing SU2 artifact: {relative}")
        _require(sha256_file(path) == expected, f"SU2 artifact SHA-256 mismatch: {relative}")
        _require(relative not in by_path, f"duplicate SU2 artifact: {relative}")
        by_path[relative] = artifact

    primary = _text(manifest, "primary_report")
    _require(primary in by_path, "primary SU2 report must be hash-bound")
    report = json.loads(_resolve(root, Path(primary)).read_text(encoding="utf-8"))
    _validate_report(report)
    provenance = manifest.get("provenance")
    _require(isinstance(provenance, dict), "SU2 release requires provenance")
    observed = report["provenance"]["candidate"]
    for key in ("candidate_sha256", "repository_commit", "tt_metal_commit"):
        _require(provenance.get(key) == observed.get(key), f"SU2 provenance mismatch: {key}")

    sessions = manifest.get("sessions")
    _require(isinstance(sessions, list) and sessions, "SU2 release requires sessions")
    ids = [session.get("id") for session in sessions if isinstance(session, dict)]
    _require(len(ids) == len(sessions) == len(set(ids)), "SU2 session IDs must be unique")
    for session in sessions:
        _require(isinstance(session, dict), "SU2 session must be an object")
        _require(session.get("performance_report") in by_path, "SU2 session report must be hash-bound")
        _require(session.get("session_manifest") in by_path, "SU2 session manifest must be hash-bound")
        session_payload = json.loads(
            _resolve(root, Path(str(session["session_manifest"]))).read_text(encoding="utf-8")
        )
        _require(session_payload.get("no_discarded_performance_runs") is True, "SU2 session must disclose no discarded runs")
        _require(session_payload["candidate"]["sha256"] == provenance["candidate_sha256"], "SU2 session candidate mismatch")

    claim = manifest.get("claim")
    _require(isinstance(claim, dict), "SU2 release requires claim")
    level = claim.get("level")
    _require(isinstance(level, int) and 0 <= level <= 3, "SU2 claim level must be 0 through 3")
    _require(claim.get("public_session_count") == len(sessions), "SU2 claim session count mismatch")
    if level == 1:
        _require(len(sessions) == 1, "SU2 Claim Level 1 requires exactly one session")
        _require(claim.get("stable_benchmark") is False, "SU2 Claim Level 1 must be non-stable")
    if level >= 2:
        _require(len(sessions) >= 3, "SU2 Claim Level 2 requires at least three independent sessions")
        _require(claim.get("stable_benchmark") is True, "SU2 Claim Level 2 requires stable_benchmark=true")
        _require(all(s.get("qualification_passed") is True for s in sessions), "every stable SU2 session must pass")

    _require(set(manifest.get("nonclaims", [])) == NONCLAIMS, "SU2 release nonclaims are incomplete")
    if not any(a.get("role") == "measured-hardware-ceiling" for a in artifacts):
        claim_surface = {key: value for key, value in manifest.items() if key not in {"nonclaims", "definitions"}}
        for value in _walk(claim_surface):
            normalized = str(value).lower().replace("_", " ").replace("-", " ")
            if "measured bandwidth" in normalized:
                raise SU2ReleaseError("measured bandwidth requires a hash-bound ceiling artifact")
    charts = manifest.get("charts")
    _require(isinstance(charts, list), "SU2 release requires charts")
    _require({chart.get("id") for chart in charts} == CHART_IDS, "SU2 release chart contract mismatch")
    for chart in charts:
        _require(str(chart.get("output", "")).endswith(".svg"), "SU2 chart output must be SVG")
    _text(manifest, "processed_output")
    _text(manifest, "raw_samples_output")


def _validate_report(report: Mapping[str, Any]) -> None:
    _require(report.get("schema") == "tt-rqm-su2-compose-report.v1", "SU2 report schema mismatch")
    _require(report.get("benchmark_stage") == "performance", "SU2 primary report must be performance")
    _require(report.get("performance_eligible") is True, "SU2 performance report is not eligible")
    _require(report.get("stable_benchmark") is False, "first SU2 report must remain non-stable")
    _require(report.get("lifecycle") == {"close_count": 1, "create_count": 1, "device_count": 1, "device_id": 0}, "SU2 lifecycle mismatch")
    results = report.get("results")
    _require(isinstance(results, list) and [(r["B"], r["K"]) for r in results] == list(CASES), "SU2 case sweep mismatch")
    for result in results:
        batch, steps = int(result["B"]), int(result["K"])
        repeats = max(1, math.ceil(2_621_440 / (batch * steps)))
        _require(result["repeat_count"] == repeats, "SU2 repeat count mismatch")
        _require(result["warmup_pairs"] == 2 and result["samples"] == 10, "SU2 timing contract mismatch")
        metadata = result["candidate_metadata"]
        _require(metadata["device_count"] == 1 and metadata["device_id"] == 0, "SU2 case device mismatch")
        _require(metadata["core_count"] == min(math.ceil(batch / 1024), 56), "SU2 core split mismatch")
        _require(metadata["fused_dispatches_per_chain"] == 1, "SU2 fused dispatch mismatch")
        _require(metadata["unfused_dispatches_per_chain"] == steps - 1, "SU2 unfused dispatch mismatch")
        _require(metadata["fused_accumulator_storage"] == "tensix_l1_ping_pong", "SU2 fused storage mismatch")
        raw = result["raw_candidate_timings_s"]
        _require(len(raw["fused_samples"]) == len(raw["unfused_samples"]) == 10, "SU2 raw sample count mismatch")
        _require(raw["paired_order"] == ["fused_first" if i % 2 == 0 else "unfused_first" for i in range(10)], "SU2 paired order mismatch")
        for path in ("fused", "unfused"):
            correctness = result[path]["correctness"]
            _require(correctness["validated_values"] == 6 * batch, "SU2 whole-output count mismatch")
            _require(correctness["failing_values"] == correctness["nonfinite_values"] == 0, "SU2 correctness failure")
            _require(len(result[path]["timing_s"]["samples"]) == 10, "SU2 normalized sample count mismatch")


def generate_release(
    manifest_path: Path = DEFAULT_MANIFEST,
    *,
    repo_root: Path | None = None,
    destination_root: Path | None = None,
) -> tuple[Path, ...]:
    root = (repo_root or Path.cwd()).resolve()
    manifest = load_manifest(_resolve(root, manifest_path))
    report = json.loads(_resolve(root, Path(manifest["primary_report"])).read_text(encoding="utf-8"))
    output_root = (destination_root or root).resolve()
    processed = Path(manifest["processed_output"])
    raw_output = Path(manifest["raw_samples_output"])
    charts = {chart["id"]: Path(chart["output"]) for chart in manifest["charts"]}
    for path in (processed, raw_output, *charts.values()):
        (output_root / path).parent.mkdir(parents=True, exist_ok=True)
    (output_root / processed).write_text(json.dumps(_processed(manifest, report), indent=2, sort_keys=True) + "\n")
    (output_root / raw_output).write_text(json.dumps(_raw_samples(report), indent=2, sort_keys=True) + "\n")
    _render(report, charts, output_root)
    return (processed, raw_output, *(charts[key] for key in sorted(charts)))


def _processed(manifest: Mapping[str, Any], report: Mapping[str, Any]) -> dict[str, Any]:
    cases = []
    for result in report["results"]:
        cases.append({
            "B": result["B"], "K": result["K"], "repeat_count": result["repeat_count"],
            "core_count": result["candidate_metadata"]["core_count"],
            "fused_median_s": result["fused"]["timing_s"]["median"],
            "unfused_median_s": result["unfused"]["timing_s"]["median"],
            "fused_over_unfused_median": result["comparison"]["fused_over_unfused_median"],
            "steps_per_s_fused": result["comparison"]["steps_per_s_fused"],
            "trajectories_per_s_fused": result["comparison"]["trajectories_per_s_fused"],
            "qmul_per_s_fused": result["comparison"]["qmul_per_s_fused"],
            "fused_logical_bytes": result["comparison"]["fused_logical_bytes"],
            "unfused_logical_bytes": result["comparison"]["unfused_logical_bytes"],
            "fused_correctness": result["fused"]["correctness"],
            "unfused_correctness": result["unfused"]["correctness"],
        })
    return {"schema": "tt-rqm-su2-compose-processed.v1", "benchmark_id": manifest["benchmark_id"],
            "claim": manifest["claim"], "source_report": manifest["primary_report"], "cases": cases}


def _raw_samples(report: Mapping[str, Any]) -> dict[str, Any]:
    return {"schema": "tt-rqm-su2-compose-raw-samples.v1", "source_report": "reports/tt_hardware_su2_compose_first_comparison.json",
            "stable_benchmark": False, "cases": [{"B": r["B"], "K": r["K"], "repeat_count": r["repeat_count"],
            "paired_order": r["raw_candidate_timings_s"]["paired_order"],
            "fused_batch_samples_s": r["raw_candidate_timings_s"]["fused_samples"],
            "unfused_batch_samples_s": r["raw_candidate_timings_s"]["unfused_samples"],
            "fused_per_chain_samples_s": r["fused"]["timing_s"]["samples"],
            "unfused_per_chain_samples_s": r["unfused"]["timing_s"]["samples"]} for r in report["results"]]}


def _render(report: Mapping[str, Any], charts: Mapping[str, Path], root: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcdefaults()
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "svg.fonttype": "path",
                                 "svg.hashsalt": "tt-rqm-wormhole-su2-compose-v1"})
    import matplotlib.pyplot as plt
    results = report["results"]
    labels = [f"{r['B']:,}x{r['K']}" for r in results]
    x = list(range(len(results)))
    blue, orange, green = "#4c78a8", "#f58518", "#54a24b"
    metadata = {"Date": None, "Creator": "tt-rqm-kernels deterministic SU2 generator"}

    fig, ax = plt.subplots(figsize=(10.2, 4.4), constrained_layout=True)
    width = 0.38
    ax.bar([v - width / 2 for v in x], [r["fused"]["timing_s"]["median"] * 1e3 for r in results], width, label="fused", color=blue)
    ax.bar([v + width / 2 for v in x], [r["unfused"]["timing_s"]["median"] * 1e3 for r in results], width, label="unfused", color=orange)
    ax.set_xticks(x, labels, rotation=25, ha="right"); ax.set_ylabel("Median per-chain latency (ms)")
    ax.set_title("Fused and unfused latency - one session, not stability"); ax.legend(); ax.grid(axis="y", alpha=.25)
    _save_svg(fig, root / charts["latency"], metadata)

    fig, ax = plt.subplots(figsize=(10.2, 4.4), constrained_layout=True)
    ax.plot(labels, [r["comparison"]["steps_per_s_fused"] / 1e6 for r in results], marker="o", color=blue)
    ax.set_ylabel("Fused evolution steps/s (millions)"); ax.set_xlabel("B x K"); ax.tick_params(axis="x", rotation=25)
    ax.set_title("Fused throughput - one session, not stability"); ax.grid(axis="y", alpha=.25)
    _save_svg(fig, root / charts["throughput"], metadata)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), constrained_layout=True)
    steps = [r["K"] for r in results]
    axes[0].semilogy(steps, [r["fused"]["correctness"]["max_abs_error"] for r in results], "o", color=blue, label="max error")
    axes[0].axhline(1e-4, linestyle="--", color=orange, label="atol"); axes[0].set_xlabel("K"); axes[0].set_ylabel("Absolute error"); axes[0].legend()
    for field, color in (("quaternion_norm_drift", blue), ("phase_norm_drift", orange), ("bloch_norm_drift", green)):
        axes[1].semilogy(steps, [r["fused"]["correctness"][field] for r in results], "o", color=color, label=field.replace("_", " "))
    axes[1].set_xlabel("K"); axes[1].set_ylabel("Drift"); axes[1].legend(fontsize=7)
    fig.suptitle("Whole-output error and drift versus chain length")
    _save_svg(fig, root / charts["error_drift"], metadata)

    fig, ax = plt.subplots(figsize=(10.2, 4.4), constrained_layout=True)
    for index, result in enumerate(results):
        paired = zip(result["fused"]["timing_s"]["samples"], result["unfused"]["timing_s"]["samples"], strict=True)
        for sample, (fused, unfused) in enumerate(paired):
            offset = (sample - 4.5) * .018
            ax.plot([index + offset - .09, index + offset + .09], [fused * 1e3, unfused * 1e3], color="#999999", alpha=.45, linewidth=.7)
            ax.scatter(index + offset - .09, fused * 1e3, s=9, color=blue)
            ax.scatter(index + offset + .09, unfused * 1e3, s=9, color=orange)
    ax.set_xticks(x, labels, rotation=25, ha="right"); ax.set_ylabel("Per-chain time (ms)")
    ax.set_title("Ten raw paired samples per case - one session, not stability"); ax.grid(axis="y", alpha=.25)
    _save_svg(fig, root / charts["raw_paired_samples"], metadata)

    fig, ax = plt.subplots(figsize=(10.2, 4.4), constrained_layout=True)
    phase_names = ("buffer_allocation", "program_build", "h2d", "warmup", "d2h")
    bottoms = [0.0] * len(results)
    colors = (blue, orange, green, "#e45756", "#72b7b2")
    for phase, color in zip(phase_names, colors, strict=True):
        values = [r["raw_candidate_timings_s"][phase] for r in results]
        ax.bar(x, values, bottom=bottoms, label=phase.replace("_", " "), color=color)
        bottoms = [a + b for a, b in zip(bottoms, values, strict=True)]
    ax.set_xticks(x, labels, rotation=25, ha="right"); ax.set_ylabel("Recorded host phase time (s)")
    ax.set_title("Collection timing breakdown"); ax.legend(fontsize=7, ncol=3)
    _save_svg(fig, root / charts["timing_breakdown"], metadata)


def _resolve(root: Path, path: Path) -> Path:
    candidate = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise SU2ReleaseError(f"path escapes repository: {path}") from exc
    return candidate


def _text(value: Mapping[str, Any], key: str) -> str:
    observed = value.get(key)
    _require(isinstance(observed, str) and bool(observed), f"SU2 release requires {key}")
    return str(observed)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SU2ReleaseError(message)


def _walk(value: Any):
    if isinstance(value, Mapping):
        for key, nested in value.items():
            yield key; yield from _walk(nested)
    elif isinstance(value, list):
        for nested in value: yield from _walk(nested)
    elif isinstance(value, str): yield value
