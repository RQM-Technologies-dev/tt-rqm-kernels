"""Subprocess runner for TT-Lang simulator StructuredBench reports."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from tt_rqm_kernels.backends.tt_lang.availability import (
    TTLangAvailability,
    TTLangSimulatorUnavailable,
    check_tt_lang_sim,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
KERNEL_DIR = REPO_ROOT / "tt_rqm_kernels" / "backends" / "tt_lang"
KERNEL_SCRIPTS = {
    "block": KERNEL_DIR / "qmul_sim_kernel.py",
    "raw": KERNEL_DIR / "qmul_raw_sim_kernel.py",
}
SUPPORTED_VARIANTS = tuple(KERNEL_SCRIPTS)


class QmulCase(Protocol):
    """Minimal case interface consumed by the TT-Lang runner."""

    workload: str
    items: int
    iterations: int
    warmup: int


def run_qmul_cases(
    cases: Iterable[QmulCase],
    *,
    seed: int,
    sim_cli: str | None = None,
    variant: str = "block",
    trace: bool = False,
    trace_output: Path | None = None,
    stats_output: Path | None = None,
    execution_label: str = "simulator",
    stable_benchmark: bool = False,
    methodology_note: str = "TT-Lang functional simulator run; not hardware performance.",
) -> dict[str, object]:
    """Run qmul cases through `tt-lang-sim` and combine their reports."""

    if variant not in KERNEL_SCRIPTS:
        raise ValueError(f"unsupported TT-Lang qmul variant: {variant}")
    if execution_label != "simulator":
        raise ValueError("TT-Lang simulator reports must use execution_label=simulator")

    availability = check_tt_lang_sim(sim_cli=sim_cli)
    if not availability.available or availability.sim_cli is None:
        raise TTLangSimulatorUnavailable(availability)

    case_list = list(cases)
    if not case_list:
        raise ValueError("at least one qmul case is required")

    trace_requested = trace or trace_output is not None or stats_output is not None
    reports: list[dict[str, object]] = []
    trace_reports: list[dict[str, object]] = []
    for index, case in enumerate(case_list):
        report, trace_report = _run_one_case(
            case,
            seed=seed + index,
            sim_cli=availability.sim_cli,
            kernel_script=KERNEL_SCRIPTS[variant],
            availability=availability,
            trace_requested=trace_requested,
            trace_output=_case_output_path(trace_output, index, len(case_list)),
            stats_output=_case_output_path(stats_output, index, len(case_list)),
        )
        reports.append(report)
        trace_reports.append(trace_report)

    first = reports[0]
    combined = {
        key: value
        for key, value in first.items()
        if key not in {"generated_at_utc", "results", "seed"}
    }
    combined.update(
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
            "execution_label": execution_label,
            "stable_benchmark": stable_benchmark,
            "methodology_note": methodology_note,
            "tt_lang_sim": _simulator_metadata(
                first,
                availability=availability,
                trace_reports=trace_reports,
            ),
            "results": [
                {
                    **result,
                    "execution_label": execution_label,
                    "stable_benchmark": stable_benchmark,
                    "methodology_note": methodology_note,
                }
                for report in reports
                for result in _report_results(report)
            ],
        }
    )
    return combined


def _simulator_metadata(
    report: dict[str, object],
    *,
    availability: TTLangAvailability,
    trace_reports: list[dict[str, object]],
) -> dict[str, object]:
    metadata = report.get("tt_lang_sim", {})
    if not isinstance(metadata, dict):
        metadata = {}
    return {
        **metadata,
        "sim_cli": Path(str(availability.sim_cli)).name,
        "sim_version": availability.version,
        "stats_cli": (
            Path(str(availability.stats_cli)).name
            if availability.stats_cli is not None
            else None
        ),
        **_combined_trace_metadata(trace_reports, availability=availability),
    }


def _run_one_case(
    case: QmulCase,
    *,
    seed: int,
    sim_cli: str,
    kernel_script: Path,
    availability: TTLangAvailability,
    trace_requested: bool,
    trace_output: Path | None,
    stats_output: Path | None,
) -> tuple[dict[str, object], dict[str, object]]:
    if case.workload != "qmul":
        raise ValueError(f"TT-Lang simulator currently supports qmul only, got {case.workload}")

    with tempfile.TemporaryDirectory(prefix="tt-rqm-ttlang-") as tmp_dir:
        output_path = Path(tmp_dir) / "report.json"
        trace_path = trace_output if trace_output is not None else Path(tmp_dir) / "trace.jsonl"
        if trace_requested and trace_output is not None:
            trace_output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sim_cli,
            str(kernel_script),
        ]
        if trace_requested:
            command.extend(["--trace", str(trace_path)])
        command.extend(
            [
                "--items",
                str(case.items),
                "--iters",
                str(case.iterations),
                "--warmup",
                str(case.warmup),
                "--seed",
                str(seed),
                "--json-output",
                str(output_path),
            ]
        )
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=_sim_env(),
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "tt-lang-sim qmul run failed\n"
                f"command: {' '.join(command)}\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        report = json.loads(output_path.read_text(encoding="utf-8"))
        trace_report = _collect_trace_stats(
            trace_requested=trace_requested,
            trace_path=trace_path,
            trace_retained=trace_output is not None,
            stats_output=stats_output,
            availability=availability,
        )
        return report, trace_report


def _collect_trace_stats(
    *,
    trace_requested: bool,
    trace_path: Path,
    trace_retained: bool,
    stats_output: Path | None,
    availability: TTLangAvailability,
) -> dict[str, object]:
    if not trace_requested:
        return {
            "trace_enabled": False,
            "trace_path": None,
            "trace_retained": False,
            "stats_available": availability.stats_available,
            "stats_summary": None,
            "stats_error": None,
        }

    if availability.stats_cli is None:
        stats_error = availability.stats_reason
        _write_optional_text(stats_output, stats_error + "\n")
        return {
            "trace_enabled": True,
            "trace_path": str(trace_path) if trace_retained else None,
            "trace_retained": trace_retained,
            "stats_available": False,
            "stats_summary": None,
            "stats_error": stats_error,
        }

    command = [availability.stats_cli, str(trace_path)]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=_sim_env(),
        )
    except OSError as exc:
        stats_error = f"tt-lang-sim-stats failed to start: {exc}"
        _write_optional_text(stats_output, stats_error + "\n")
        return {
            "trace_enabled": True,
            "trace_path": str(trace_path) if trace_retained else None,
            "trace_retained": trace_retained,
            "stats_available": True,
            "stats_summary": None,
            "stats_error": stats_error,
        }

    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    if completed.returncode != 0:
        detail = stderr or stdout or "no output"
        stats_error = (
            f"tt-lang-sim-stats failed with exit code {completed.returncode}: "
            f"{_compact_text(detail)}"
        )
        _write_optional_text(stats_output, stats_error + "\n")
        return {
            "trace_enabled": True,
            "trace_path": str(trace_path) if trace_retained else None,
            "trace_retained": trace_retained,
            "stats_available": True,
            "stats_summary": None,
            "stats_error": stats_error,
        }

    stats_summary = stdout or stderr
    _write_optional_text(stats_output, stats_summary + ("\n" if stats_summary else ""))
    return {
        "trace_enabled": True,
        "trace_path": str(trace_path) if trace_retained else None,
        "trace_retained": trace_retained,
        "stats_available": True,
        "stats_summary": stats_summary,
        "stats_error": None,
    }


def _combined_trace_metadata(
    trace_reports: list[dict[str, object]],
    *,
    availability: TTLangAvailability,
) -> dict[str, object]:
    if not trace_reports:
        return {
            "trace_enabled": False,
            "trace_path": None,
            "trace_retained": False,
            "stats_available": availability.stats_available,
            "stats_summary": None,
            "stats_error": None,
        }
    if len(trace_reports) == 1:
        return trace_reports[0]

    summaries = [
        str(report["stats_summary"])
        for report in trace_reports
        if report.get("stats_summary")
    ]
    errors = [
        str(report["stats_error"])
        for report in trace_reports
        if report.get("stats_error")
    ]
    trace_paths = [
        report.get("trace_path")
        for report in trace_reports
        if report.get("trace_path") is not None
    ]
    return {
        "trace_enabled": any(bool(report.get("trace_enabled")) for report in trace_reports),
        "trace_path": trace_paths,
        "trace_retained": any(bool(report.get("trace_retained")) for report in trace_reports),
        "stats_available": availability.stats_available,
        "stats_summary": "\n\n".join(summaries) if summaries else None,
        "stats_error": "\n".join(errors) if errors else None,
    }


def _case_output_path(path: Path | None, index: int, total: int) -> Path | None:
    if path is None:
        return None
    if total == 1:
        return path
    suffix = path.suffix or ".txt"
    return path.with_name(f"{path.stem}_case{index}{suffix}")


def _write_optional_text(path: Path | None, content: str) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _compact_text(value: str, *, limit: int = 1000) -> str:
    compacted = " ".join(value.split())
    if len(compacted) <= limit:
        return compacted
    return compacted[: limit - 3] + "..."


def _report_results(report: dict[str, object]) -> list[dict[str, object]]:
    results = report["results"]
    if not isinstance(results, list):
        raise TypeError("report results must be a list")
    for result in results:
        if not isinstance(result, dict):
            raise TypeError("each report result must be a dict")
    return results


def _sim_env() -> dict[str, str]:
    env = os.environ.copy()
    repo = str(REPO_ROOT)
    existing = env.get("PYTHONPATH")
    if existing:
        parts = existing.split(os.pathsep)
        if repo not in parts:
            env["PYTHONPATH"] = os.pathsep.join([repo, existing])
    else:
        env["PYTHONPATH"] = repo
    return env
