"""StructuredBench benchmark CLI for structured quaternion tensor kernels."""

from __future__ import annotations

import argparse
from array import array
import json
import math
import os
import platform
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal

import torch

from tt_rqm_kernels.backends import scalar_reference, torch_backend
from tt_rqm_kernels.backends.tt_lang.availability import TTLangSimulatorUnavailable
from tt_rqm_kernels.benchmark_integrity import (
    BenchmarkStage,
    command_sha256,
    independent_qmul_golden,
    repository_commit,
    timing_summary,
    validate_execution_policy,
    validate_external_metrics,
    validate_qmul_output,
    validate_report,
)

SCHEMA_VERSION = "structuredbench.v1"
EXTERNAL_QMUL_PROTOCOL = "tt-rqm-external-qmul.v1"
SUPPORTED_SUITES = ("smoke", "full", "qmul", "qrotate")
SUPPORTED_BACKENDS = ("torch", "tt-lang-sim", "external-qmul")
ExecutionLabel = Literal["cpu", "simulator", "emulation", "hardware"]
EXECUTION_LABELS: tuple[ExecutionLabel, ...] = (
    "cpu",
    "simulator",
    "emulation",
    "hardware",
)
SUPPORTED_DTYPES = {
    "float32": torch.float32,
    "float64": torch.float64,
}
SCALAR_CHECK_SAMPLES = 8
SCALAR_ERROR_TOLERANCES = {
    "float32": 1e-4,
    "float64": 1e-9,
}


@dataclass(frozen=True)
class BenchmarkCase:
    """One benchmark configuration."""

    workload: str
    items: int
    iterations: int
    warmup: int
    throughput_unit: str


@dataclass(frozen=True)
class BenchmarkResult:
    """One measured benchmark result."""

    suite: str
    workload: str
    backend: str
    device: str
    dtype: str
    items: int
    iterations: int
    warmup: int
    structured_shape: str
    throughput_unit: str
    elapsed_s: float
    latency_ms: float
    throughput: float
    max_abs_error: float
    max_rel_error: float
    rms_error: float
    stability_max_abs: float | None
    scalar_reference_max_abs_error: float | None
    estimated_flops: int
    estimated_flops_per_s: float
    estimated_bytes_read: int
    estimated_bytes_written: int
    estimated_total_bytes: int
    effective_gb_per_s: float
    arithmetic_intensity_flops_per_byte: float
    checksum: float
    execution_label: ExecutionLabel = "cpu"
    stable_benchmark: bool = False
    methodology_note: str | None = None
    correctness: dict[str, object] | None = None
    timing: dict[str, object] | None = None
    provenance: dict[str, object] | None = None
    implementation_class: str | None = None
    performance_eligible: bool = False


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def build_cases(
    suite: str,
    *,
    items_override: int | None = None,
    iterations_override: int | None = None,
    warmup_override: int | None = None,
) -> list[BenchmarkCase]:
    """Build benchmark cases for a named suite."""

    if suite == "smoke":
        cases = [
            BenchmarkCase("qmul", 1024, 5, 2, "qmul/s"),
            BenchmarkCase("qrotate", 1024, 5, 2, "rotations/s"),
            BenchmarkCase("qnormalize", 1024, 5, 2, "normalizations/s"),
            BenchmarkCase("qinverse", 1024, 5, 2, "inverses/s"),
            BenchmarkCase("phase_update", 2048, 5, 2, "phase-updates/s"),
        ]
    elif suite == "qmul":
        cases = [
            BenchmarkCase("qmul", 4096, 30, 5, "qmul/s"),
            BenchmarkCase("qmul", 65536, 30, 5, "qmul/s"),
            BenchmarkCase("qmul", 262144, 30, 5, "qmul/s"),
        ]
    elif suite == "qrotate":
        cases = [
            BenchmarkCase("qrotate", 4096, 20, 5, "rotations/s"),
            BenchmarkCase("qrotate", 65536, 20, 5, "rotations/s"),
            BenchmarkCase("qrotate", 262144, 20, 5, "rotations/s"),
        ]
    elif suite == "full":
        cases = [
            BenchmarkCase("qmul", 4096, 30, 5, "qmul/s"),
            BenchmarkCase("qmul", 65536, 30, 5, "qmul/s"),
            BenchmarkCase("qmul", 262144, 30, 5, "qmul/s"),
            BenchmarkCase("qrotate", 4096, 20, 5, "rotations/s"),
            BenchmarkCase("qrotate", 65536, 20, 5, "rotations/s"),
            BenchmarkCase("qrotate", 262144, 20, 5, "rotations/s"),
            BenchmarkCase("qnormalize", 4096, 30, 5, "normalizations/s"),
            BenchmarkCase("qnormalize", 65536, 30, 5, "normalizations/s"),
            BenchmarkCase("qnormalize", 262144, 30, 5, "normalizations/s"),
            BenchmarkCase("qinverse", 4096, 20, 5, "inverses/s"),
            BenchmarkCase("qinverse", 65536, 20, 5, "inverses/s"),
            BenchmarkCase("qinverse", 262144, 20, 5, "inverses/s"),
            BenchmarkCase("phase_update", 8192, 30, 5, "phase-updates/s"),
            BenchmarkCase("phase_update", 131072, 30, 5, "phase-updates/s"),
            BenchmarkCase("phase_update", 524288, 30, 5, "phase-updates/s"),
        ]
    else:
        raise ValueError(f"unsupported suite: {suite}")

    return [
        BenchmarkCase(
            workload=case.workload,
            items=items_override or case.items,
            iterations=iterations_override or case.iterations,
            warmup=case.warmup if warmup_override is None else warmup_override,
            throughput_unit=case.throughput_unit,
        )
        for case in cases
    ]


def run_suite(
    suite: str,
    *,
    backend: str = "torch",
    device_name: str = "cpu",
    dtype_name: str = "float32",
    seed: int = 0,
    items_override: int | None = None,
    iterations_override: int | None = None,
    warmup_override: int | None = None,
    external_command: str | None = None,
    sim_cli: str | None = None,
    tt_lang_variant: str = "block",
    tt_lang_trace: bool = False,
    tt_lang_trace_output: Path | None = None,
    tt_lang_stats_output: Path | None = None,
    execution_label: ExecutionLabel | None = None,
    stable_benchmark: bool = False,
    methodology_note: str | None = None,
    repetitions: int = 1,
    benchmark_stage: BenchmarkStage | None = None,
) -> dict[str, object]:
    """Run a StructuredBench suite and return a JSON-serializable report."""

    if dtype_name not in SUPPORTED_DTYPES:
        raise ValueError(f"unsupported dtype: {dtype_name}")
    resolved_execution_label = _resolve_execution_label(backend, execution_label)
    resolved_methodology_note = _methodology_note(
        backend,
        resolved_execution_label,
        methodology_note,
    )

    if backend == "tt-lang-sim":
        validate_execution_policy(
            backend=backend,
            execution_label=resolved_execution_label,
            stable_benchmark=stable_benchmark,
            items=[items_override or 128],
        )
        report = _run_tt_lang_sim_suite(
            suite,
            dtype_name=dtype_name,
            seed=seed,
            items_override=items_override,
            iterations_override=iterations_override,
            warmup_override=warmup_override,
            sim_cli=sim_cli,
            variant=tt_lang_variant,
            trace=tt_lang_trace,
            trace_output=tt_lang_trace_output,
            stats_output=tt_lang_stats_output,
            execution_label=resolved_execution_label,
            stable_benchmark=stable_benchmark,
            methodology_note=resolved_methodology_note,
        )
        validate_report(report)
        return report
    if backend == "external-qmul":
        report = _run_external_qmul_suite(
            suite,
            dtype_name=dtype_name,
            seed=seed,
            items_override=items_override,
            iterations_override=iterations_override,
            warmup_override=warmup_override,
            external_command=external_command,
            execution_label=resolved_execution_label,
            stable_benchmark=stable_benchmark,
            methodology_note=resolved_methodology_note,
            repetitions=repetitions,
            benchmark_stage=benchmark_stage,
        )
        validate_report(report)
        return report
    if backend != "torch":
        raise ValueError(f"unsupported backend: {backend}")

    validate_execution_policy(
        backend=backend,
        execution_label=resolved_execution_label,
        stable_benchmark=stable_benchmark,
    )

    device = _resolve_device(device_name)
    dtype = SUPPORTED_DTYPES[dtype_name]
    cases = build_cases(
        suite,
        items_override=items_override,
        iterations_override=iterations_override,
        warmup_override=warmup_override,
    )

    results = [
        _run_case(
            suite=suite,
            case=case,
            backend=backend,
            device=device,
            dtype=dtype,
            dtype_name=dtype_name,
            seed=seed + index,
            execution_label=resolved_execution_label,
            stable_benchmark=stable_benchmark,
            methodology_note=resolved_methodology_note,
        )
        for index, case in enumerate(cases)
    ]

    report = {
        "schema": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "suite": suite,
        "backend": backend,
        "device": str(device),
        "execution_label": resolved_execution_label,
        "stable_benchmark": stable_benchmark,
        "methodology_note": resolved_methodology_note,
        "dtype": dtype_name,
        "seed": seed,
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "results": [asdict(result) for result in results],
        "repetitions": 1,
        "case_items": [case.items for case in cases],
    }
    validate_report(report)
    return report


def render_table(report: dict[str, object]) -> str:
    """Render a report as a compact plain-text table."""

    results = report["results"]
    if not isinstance(results, list):
        raise TypeError("report results must be a list")

    lines = [
        "StructuredBench",
        (
            f"schema={report['schema']} suite={report['suite']} "
            f"backend={report['backend']} device={report['device']} "
            f"execution={report.get('execution_label', 'unknown')} "
            f"stable={str(report.get('stable_benchmark', False)).lower()} "
            f"dtype={report['dtype']}"
        ),
        "",
    ]
    headers = [
        "workload",
        "items",
        "iters",
        "latency_ms",
        "throughput",
        "unit",
        "max_abs_err",
        "rms_err",
        "stability",
        "scalar_ref",
    ]
    rows = []
    for result in results:
        if not isinstance(result, dict):
            raise TypeError("each result must be a dict")
        stability = result["stability_max_abs"]
        scalar_ref = result["scalar_reference_max_abs_error"]
        rows.append(
            [
                str(result["workload"]),
                str(result["items"]),
                str(result["iterations"]),
                f"{float(result['latency_ms']):.4f}",
                f"{float(result['throughput']):.2f}",
                str(result["throughput_unit"]),
                f"{float(result['max_abs_error']):.3e}",
                f"{float(result['rms_error']):.3e}",
                "-" if stability is None else f"{float(stability):.3e}",
                "-" if scalar_ref is None else f"{float(scalar_ref):.3e}",
            ]
        )

    widths = [
        max(len(headers[column]), *(len(row[column]) for row in rows))
        for column in range(len(headers))
    ]
    lines.append(_format_row(headers, widths))
    lines.append(_format_row(["-" * width for width in widths], widths))
    lines.extend(_format_row(row, widths) for row in rows)
    return "\n".join(lines)


def render_markdown_report(report: dict[str, object]) -> str:
    """Render a StructuredBench report as Markdown."""

    results = _report_results(report)
    benchmark_rows = [
        [
            str(result["workload"]),
            str(result["items"]),
            str(result["iterations"]),
            f"{float(result['latency_ms']):.4f}",
            f"{float(result['throughput']):.2f}",
            str(result["throughput_unit"]),
            f"{float(result['max_abs_error']):.3e}",
            f"{float(result['rms_error']):.3e}",
            _optional_scientific(result["stability_max_abs"]),
            _optional_scientific(result["scalar_reference_max_abs_error"]),
        ]
        for result in results
    ]
    hardware_rows = [
        [
            str(result["workload"]),
            str(result["items"]),
            str(result["estimated_flops"]),
            f"{float(result['estimated_flops_per_s']):.3e}",
            str(result["estimated_total_bytes"]),
            f"{float(result['effective_gb_per_s']):.3f}",
            f"{float(result['arithmetic_intensity_flops_per_byte']):.3f}",
        ]
        for result in results
    ]
    integrity_rows = []
    for result in results:
        correctness = result.get("correctness") or {}
        timing = result.get("timing") or {}
        device_timing = timing.get("device_s", {}) if isinstance(timing, dict) else {}
        end_to_end = timing.get("end_to_end_s", {}) if isinstance(timing, dict) else {}
        integrity_rows.append(
            [
                str(result["workload"]),
                str(result.get("implementation_class") or "-"),
                str(bool(result.get("performance_eligible", False))).lower(),
                str(correctness.get("passed", False)).lower(),
                str(correctness.get("validated_values", "-")),
                f"{float(correctness.get('whole_output_max_abs_error', 0.0)):.3e}",
                str(timing.get("repetitions", 1)) if isinstance(timing, dict) else "1",
                _optional_scientific(device_timing.get("median")),
                _optional_scientific(device_timing.get("p95")),
                _optional_scientific(end_to_end.get("median")),
            ]
        )

    return "\n".join(
        [
            "# StructuredBench Report",
            "",
            f"Generated: `{report['generated_at_utc']}`",
            "",
            (
                f"Backend: `{report['backend']}`  "
                f"Device: `{report['device']}`  "
                f"Execution: `{report.get('execution_label', 'unknown')}`  "
                f"Stable benchmark: `{str(report.get('stable_benchmark', False)).lower()}`  "
                f"Dtype: `{report['dtype']}`  "
                f"Suite: `{report['suite']}`"
            ),
            "",
            *_report_intro(report),
            "## Benchmark Results",
            "",
            _markdown_table(
                [
                    "workload",
                    "items",
                    "iters",
                    "latency_ms",
                    "throughput",
                    "unit",
                    "max_abs_err",
                    "rms_err",
                    "stability",
                    "scalar_ref",
                ],
                benchmark_rows,
            ),
            "",
            "## Conformance and Timing Integrity",
            "",
            _markdown_table(
                [
                    "workload",
                    "implementation_class",
                    "performance_eligible",
                    "correctness_passed",
                    "validated_values",
                    "whole_output_max_abs_err",
                    "repetitions",
                    "device_median_s",
                    "device_p95_s",
                    "end_to_end_median_s",
                ],
                integrity_rows,
            ),
            "",
            "## Hardware-Relevant Metrics",
            "",
            _markdown_table(
                [
                    "workload",
                    "items",
                    "estimated_flops",
                    "estimated_flops_per_s",
                    "estimated_total_bytes",
                    "effective_gb_per_s",
                    "arithmetic_intensity",
                ],
                hardware_rows,
            ),
            "",
            *_tt_lang_trace_stats_section(report),
            "## Notes",
            "",
            _methodology_note_line(report),
            _backend_note(report),
            _committed_report_note(report),
            *_backend_metadata_notes(report),
            "- Scalar reference checks are small deterministic spot checks used as an independent correctness contract.",
            "- FLOP and byte counts are simple documented estimates for backend comparison, not hardware-counter measurements.",
            "- Phase update includes transcendental-heavy sin/cos state generation; its FLOP estimate counts each transcendental call as one reported operation.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run StructuredBench quaternion, rotor, and phase tensor benchmarks."
    )
    parser.add_argument("--suite", choices=SUPPORTED_SUITES, default="smoke")
    parser.add_argument("--backend", choices=SUPPORTED_BACKENDS, default="torch")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=tuple(SUPPORTED_DTYPES), default="float32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--items", type=positive_int, default=None)
    parser.add_argument("--iters", type=positive_int, default=None)
    parser.add_argument("--warmup", type=nonnegative_int, default=None)
    parser.add_argument("--repetitions", type=positive_int, default=1)
    parser.add_argument(
        "--benchmark-stage",
        choices=("conformance", "performance"),
        default=None,
    )
    parser.add_argument("--threads", type=positive_int, default=None)
    parser.add_argument("--format", choices=("table", "json"), default="table")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--markdown-output", type=Path, default=None)
    parser.add_argument(
        "--execution-label",
        choices=EXECUTION_LABELS,
        default=None,
        help=(
            "Execution environment label for report metadata. Torch defaults to "
            "cpu, tt-lang-sim defaults to simulator, and external-qmul can be "
            "labeled cpu, emulation, or hardware by the caller."
        ),
    )
    parser.add_argument(
        "--stable-benchmark",
        action="store_true",
        help=(
            "Mark the report as a stable benchmark. Default is false because "
            "reference, simulator, emulation, and first hardware samples are "
            "usually methodology checks."
        ),
    )
    parser.add_argument(
        "--methodology-note",
        default=None,
        help="Optional short note describing the measurement methodology.",
    )
    parser.add_argument(
        "--external-command",
        default=None,
        help=(
            "Command used by --backend external-qmul. StructuredBench exposes "
            "TT_RQM_EXTERNAL_QMUL_DIR and TT_RQM_EXTERNAL_QMUL_MANIFEST."
        ),
    )
    parser.add_argument(
        "--sim-cli",
        default=None,
        help="Override the tt-lang-sim executable used by --backend tt-lang-sim.",
    )
    parser.add_argument(
        "--tt-lang-variant",
        choices=("block", "raw"),
        default="block",
        help="TT-Lang qmul simulator variant. Requires --backend tt-lang-sim.",
    )
    parser.add_argument(
        "--tt-lang-trace",
        action="store_true",
        help=(
            "Enable TT-Lang simulator trace capture. Uses a temporary trace file "
            "unless --tt-lang-trace-output is provided. Requires --backend tt-lang-sim."
        ),
    )
    parser.add_argument(
        "--tt-lang-trace-output",
        type=Path,
        default=None,
        help="Write the TT-Lang simulator JSONL trace to this path.",
    )
    parser.add_argument(
        "--tt-lang-stats-output",
        type=Path,
        default=None,
        help=(
            "Write tt-lang-sim-stats text output when trace capture is enabled. "
            "Also enables trace capture. Requires --backend tt-lang-sim."
        ),
    )
    args = parser.parse_args(argv)

    tt_lang_trace_args_used = (
        args.tt_lang_variant != "block"
        or args.tt_lang_trace
        or args.tt_lang_trace_output is not None
        or args.tt_lang_stats_output is not None
    )
    if tt_lang_trace_args_used and args.backend != "tt-lang-sim":
        parser.error("TT-Lang variant/trace/stat flags require --backend tt-lang-sim")

    if args.threads is not None:
        torch.set_num_threads(args.threads)

    try:
        report = run_suite(
            args.suite,
            backend=args.backend,
            device_name=args.device,
            dtype_name=args.dtype,
            seed=args.seed,
            items_override=args.items,
            iterations_override=args.iters,
            warmup_override=args.warmup,
            external_command=args.external_command,
            sim_cli=args.sim_cli,
            tt_lang_variant=args.tt_lang_variant,
            tt_lang_trace=args.tt_lang_trace,
            tt_lang_trace_output=args.tt_lang_trace_output,
            tt_lang_stats_output=args.tt_lang_stats_output,
            execution_label=args.execution_label,
            stable_benchmark=args.stable_benchmark,
            methodology_note=args.methodology_note,
            repetitions=args.repetitions,
            benchmark_stage=args.benchmark_stage,
        )
    except TTLangSimulatorUnavailable as exc:
        print(str(exc), file=sys.stderr)
        return 2
    rendered = (
        json.dumps(report, indent=2, sort_keys=True)
        if args.format == "json"
        else render_table(report)
    )

    if args.output is not None:
        _write_text(args.output, rendered + "\n")
    if args.json_output is not None:
        _write_text(args.json_output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.markdown_output is not None:
        _write_text(args.markdown_output, render_markdown_report(report))

    print(rendered)
    return 0


def _resolve_execution_label(
    backend: str,
    requested: ExecutionLabel | None,
) -> ExecutionLabel:
    default = _default_execution_label(backend)
    if requested is None:
        return default
    if backend == "torch" and requested != "cpu":
        raise ValueError("torch backend reports must use execution_label=cpu")
    if backend == "tt-lang-sim" and requested != "simulator":
        raise ValueError(
            "tt-lang-sim backend reports must use execution_label=simulator"
        )
    if backend == "external-qmul" and requested == "simulator":
        raise ValueError(
            "external-qmul reports should use cpu, emulation, or hardware; "
            "use tt-lang-sim for simulator reports"
        )
    return requested


def _default_execution_label(backend: str) -> ExecutionLabel:
    if backend == "tt-lang-sim":
        return "simulator"
    return "cpu"


def _methodology_note(
    backend: str,
    execution_label: ExecutionLabel,
    requested: str | None,
) -> str:
    if requested:
        return requested
    if backend == "tt-lang-sim":
        return "TT-Lang functional simulator run; not hardware performance."
    if backend == "external-qmul":
        return (
            f"external-qmul candidate run labeled {execution_label}; "
            "hardware claims depend on the external command and environment."
        )
    return "CPU/PyTorch reference run; not a hardware performance result."


def _resolve_device(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but torch.cuda.is_available() is false")
    return device


def _run_tt_lang_sim_suite(
    suite: str,
    *,
    dtype_name: str,
    seed: int,
    items_override: int | None,
    iterations_override: int | None,
    warmup_override: int | None,
    sim_cli: str | None,
    variant: str,
    trace: bool,
    trace_output: Path | None,
    stats_output: Path | None,
    execution_label: ExecutionLabel,
    stable_benchmark: bool,
    methodology_note: str,
) -> dict[str, object]:
    if suite != "qmul":
        raise ValueError("tt-lang-sim backend currently supports --suite qmul only")
    if dtype_name != "float32":
        raise ValueError("tt-lang-sim backend currently supports --dtype float32 only")

    case = BenchmarkCase(
        workload="qmul",
        items=items_override or 128,
        iterations=iterations_override or 1,
        warmup=0 if warmup_override is None else warmup_override,
        throughput_unit="qmul/s",
    )

    from tt_rqm_kernels.backends.tt_lang.runner import run_qmul_cases

    return run_qmul_cases(
        [case],
        seed=seed,
        sim_cli=sim_cli,
        variant=variant,
        trace=trace,
        trace_output=trace_output,
        stats_output=stats_output,
        execution_label=execution_label,
        stable_benchmark=stable_benchmark,
        methodology_note=methodology_note,
    )


def _run_external_qmul_suite(
    suite: str,
    *,
    dtype_name: str,
    seed: int,
    items_override: int | None,
    iterations_override: int | None,
    warmup_override: int | None,
    external_command: str | None,
    execution_label: ExecutionLabel,
    stable_benchmark: bool,
    methodology_note: str,
    repetitions: int,
    benchmark_stage: BenchmarkStage | None,
) -> dict[str, object]:
    if suite != "qmul":
        raise ValueError("external-qmul backend currently supports --suite qmul only")
    if dtype_name != "float32":
        raise ValueError("external-qmul backend currently supports --dtype float32 only")
    if not external_command:
        raise ValueError("--backend external-qmul requires --external-command")

    cases = _build_external_qmul_cases(
        suite,
        items_override=items_override,
        iterations_override=iterations_override,
        warmup_override=warmup_override,
        benchmark_stage=benchmark_stage,
    )
    validate_execution_policy(
        backend="external-qmul",
        execution_label=execution_label,
        stable_benchmark=stable_benchmark,
        command=external_command,
        stage=benchmark_stage,
        repetitions=repetitions,
        items=[case.items for case in cases],
        iterations=[case.iterations for case in cases],
        warmups=[case.warmup for case in cases],
    )
    results = [
        _run_external_qmul_case(
            suite=suite,
            case=case,
            external_command=external_command,
            seed=seed + index,
            execution_label=execution_label,
            stable_benchmark=stable_benchmark,
            methodology_note=methodology_note,
            repetitions=repetitions,
            benchmark_stage=benchmark_stage,
        )
        for index, case in enumerate(cases)
    ]
    device = results[0].device if results else "external-command"
    return {
        "schema": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "suite": suite,
        "backend": "external-qmul",
        "device": device,
        "execution_label": execution_label,
        "stable_benchmark": stable_benchmark,
        "methodology_note": methodology_note,
        "dtype": dtype_name,
        "seed": seed,
        "protocol": EXTERNAL_QMUL_PROTOCOL,
        "external_command": external_command,
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "results": [asdict(result) for result in results],
        "repetitions": repetitions,
        "benchmark_stage": benchmark_stage,
        "case_items": [case.items for case in cases],
        "provenance": {
            "repository_commit": repository_commit(Path(__file__).resolve().parents[1]),
            "candidate_sha256": command_sha256(
                external_command, Path(__file__).resolve().parents[1]
            ),
            "candidate": results[0].provenance if results else None,
        },
    }


def _build_external_qmul_cases(
    suite: str,
    *,
    items_override: int | None,
    iterations_override: int | None,
    warmup_override: int | None,
    benchmark_stage: BenchmarkStage | None,
) -> list[BenchmarkCase]:
    if benchmark_stage == "conformance":
        return [
            BenchmarkCase(
                "qmul",
                128 if items_override is None else items_override,
                1 if iterations_override is None else iterations_override,
                0 if warmup_override is None else warmup_override,
                "qmul/s",
            )
        ]
    return build_cases(
        suite,
        items_override=items_override,
        iterations_override=iterations_override,
        warmup_override=warmup_override,
    )


def _run_external_qmul_case(
    *,
    suite: str,
    case: BenchmarkCase,
    external_command: str,
    seed: int,
    execution_label: ExecutionLabel,
    stable_benchmark: bool,
    methodology_note: str,
    repetitions: int,
    benchmark_stage: BenchmarkStage | None,
) -> BenchmarkResult:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    a64 = torch_backend.qnormalize(_randn((case.items, 4), generator))
    b64 = torch_backend.qnormalize(_randn((case.items, 4), generator))
    a = a64.to(dtype=torch.float32)
    b = b64.to(dtype=torch.float32)
    reference = independent_qmul_golden(a, b)

    setup_samples: list[float] = []
    device_samples: list[float] = []
    end_to_end_samples: list[float] = []
    correctness: dict[str, object] | None = None
    metrics: dict[str, object] = {}
    output: torch.Tensor | None = None
    repo_root = Path(__file__).resolve().parents[1]
    candidate_hash = command_sha256(external_command, repo_root)

    with tempfile.TemporaryDirectory(prefix="tt-rqm-external-qmul-") as tmp_dir:
        work_dir = Path(tmp_dir)
        a_path = work_dir / "a.bin"
        b_path = work_dir / "b.bin"
        out_path = work_dir / "out.bin"
        metrics_path = work_dir / "metrics.json"
        manifest_path = work_dir / "manifest.json"

        _write_float32_binary(a_path, a)
        _write_float32_binary(b_path, b)
        _write_text(
            manifest_path,
            json.dumps(
                {
                    "schema": EXTERNAL_QMUL_PROTOCOL,
                    "workload": "qmul",
                    "dtype": "float32",
                    "lane_order": ["real", "i", "j", "k"],
                    "items": case.items,
                    "iterations": case.iterations,
                    "warmup": case.warmup,
                    "shape": [case.items, 4],
                    "input_format": "raw little-endian float32 row-major",
                    "output_format": "raw little-endian float32 row-major",
                    "inputs": {
                        "a": "a.bin",
                        "b": "b.bin",
                    },
                    "outputs": {
                        "out": "out.bin",
                        "metrics": "metrics.json",
                    },
                    "seed": seed,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for _ in range(repetitions):
            host_elapsed_s = _run_external_command(
                external_command,
                work_dir=work_dir,
                manifest_path=manifest_path,
                execution_label=execution_label,
            )
            metrics = _load_external_metrics(metrics_path)
            timing = validate_external_metrics(
                metrics,
                manifest,
                execution_label=execution_label,
                host_end_to_end_s=host_elapsed_s,
                candidate_sha256=candidate_hash,
                stage=benchmark_stage,
            )
            output = _read_float32_binary(out_path, (case.items, 4))
            _, correctness = validate_qmul_output(output, a, b)
            setup_samples.append(float(timing["setup_s"]))
            device_samples.append(float(timing["device_s"]))
            end_to_end_samples.append(float(timing["end_to_end_s"]))

    if output is None or correctness is None:
        raise RuntimeError("external-qmul did not produce a validated output")
    elapsed_s = float(timing_summary(device_samples)["median"])
    scalar_error = float(correctness["scalar_first_eight_max_abs_error"])
    return _result_from_output(
        suite,
        case,
        "external-qmul",
        str(metrics.get("device", "external-command")),
        torch.float32,
        "float32",
        output,
        reference,
        elapsed_s,
        scalar_reference_max_abs_error=scalar_error,
        execution_label=execution_label,
        stable_benchmark=stable_benchmark,
        methodology_note=methodology_note,
        correctness=correctness,
        timing={
            "repetitions": repetitions,
            "setup_s": timing_summary(setup_samples),
            "device_s": timing_summary(device_samples),
            "end_to_end_s": timing_summary(end_to_end_samples),
            "primary_elapsed": "device_s.median",
        },
        provenance={
            **(
                dict(metrics.get("provenance", {}))
                if isinstance(metrics.get("provenance"), dict)
                else {}
            ),
            "repository_commit": repository_commit(repo_root),
            "candidate_sha256": candidate_hash,
            "external_command": external_command,
        },
        implementation_class=str(metrics["implementation_class"]),
        performance_eligible=bool(metrics["performance_eligible"]),
    )


def _run_case(
    *,
    suite: str,
    case: BenchmarkCase,
    backend: str,
    device: torch.device,
    dtype: torch.dtype,
    dtype_name: str,
    seed: int,
    execution_label: ExecutionLabel,
    stable_benchmark: bool,
    methodology_note: str,
) -> BenchmarkResult:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if case.workload == "qmul":
        return _run_qmul(
            suite,
            case,
            backend,
            device,
            dtype,
            dtype_name,
            generator,
            execution_label,
            stable_benchmark,
            methodology_note,
        )
    if case.workload == "qrotate":
        return _run_qrotate(
            suite,
            case,
            backend,
            device,
            dtype,
            dtype_name,
            generator,
            execution_label,
            stable_benchmark,
            methodology_note,
        )
    if case.workload == "qnormalize":
        return _run_qnormalize(
            suite,
            case,
            backend,
            device,
            dtype,
            dtype_name,
            generator,
            execution_label,
            stable_benchmark,
            methodology_note,
        )
    if case.workload == "qinverse":
        return _run_qinverse(
            suite,
            case,
            backend,
            device,
            dtype,
            dtype_name,
            generator,
            execution_label,
            stable_benchmark,
            methodology_note,
        )
    if case.workload == "phase_update":
        return _run_phase_update(
            suite,
            case,
            backend,
            device,
            dtype,
            dtype_name,
            generator,
            execution_label,
            stable_benchmark,
            methodology_note,
        )
    raise ValueError(f"unsupported workload: {case.workload}")


def _run_qmul(
    suite: str,
    case: BenchmarkCase,
    backend: str,
    device: torch.device,
    dtype: torch.dtype,
    dtype_name: str,
    generator: torch.Generator,
    execution_label: ExecutionLabel,
    stable_benchmark: bool,
    methodology_note: str,
) -> BenchmarkResult:
    a64 = torch_backend.qnormalize(_randn((case.items, 4), generator))
    b64 = torch_backend.qnormalize(_randn((case.items, 4), generator))
    a = _to_device_dtype(a64, device, dtype)
    b = _to_device_dtype(b64, device, dtype)

    output, elapsed_s = _measure(
        lambda: torch_backend.qmul(a, b),
        case.warmup,
        case.iterations,
        device,
    )
    reference = torch_backend.qmul(a64, b64)
    scalar_error = _scalar_check_qmul(output, a64, b64, dtype_name)
    return _result_from_output(
        suite,
        case,
        backend,
        device,
        dtype,
        dtype_name,
        output,
        reference,
        elapsed_s,
        scalar_reference_max_abs_error=scalar_error,
        execution_label=execution_label,
        stable_benchmark=stable_benchmark,
        methodology_note=methodology_note,
    )


def _run_qrotate(
    suite: str,
    case: BenchmarkCase,
    backend: str,
    device: torch.device,
    dtype: torch.dtype,
    dtype_name: str,
    generator: torch.Generator,
    execution_label: ExecutionLabel,
    stable_benchmark: bool,
    methodology_note: str,
) -> BenchmarkResult:
    rotors64 = torch_backend.qnormalize(_randn((case.items, 4), generator))
    vectors64 = _randn((case.items, 3), generator)
    rotors = _to_device_dtype(rotors64, device, dtype)
    vectors = _to_device_dtype(vectors64, device, dtype)

    output, elapsed_s = _measure(
        lambda: torch_backend.qrotate_vector(rotors, vectors, assume_unit=True),
        case.warmup,
        case.iterations,
        device,
    )
    reference = torch_backend.qrotate_vector(rotors64, vectors64, assume_unit=True)
    stability = _max_abs(
        torch.linalg.vector_norm(output.detach().cpu().to(torch.float64), dim=-1)
        - torch.linalg.vector_norm(vectors64, dim=-1)
    )
    scalar_error = _scalar_check_qrotate(output, rotors64, vectors64, dtype_name)
    return _result_from_output(
        suite,
        case,
        backend,
        device,
        dtype,
        dtype_name,
        output,
        reference,
        elapsed_s,
        stability_max_abs=stability,
        scalar_reference_max_abs_error=scalar_error,
        execution_label=execution_label,
        stable_benchmark=stable_benchmark,
        methodology_note=methodology_note,
    )


def _run_qnormalize(
    suite: str,
    case: BenchmarkCase,
    backend: str,
    device: torch.device,
    dtype: torch.dtype,
    dtype_name: str,
    generator: torch.Generator,
    execution_label: ExecutionLabel,
    stable_benchmark: bool,
    methodology_note: str,
) -> BenchmarkResult:
    values64 = _scaled_quaternions(case.items, generator, min_scale=1e-3, max_scale=1e3)
    values = _to_device_dtype(values64, device, dtype)

    output, elapsed_s = _measure(
        lambda: torch_backend.qnormalize(values),
        case.warmup,
        case.iterations,
        device,
    )
    reference = torch_backend.qnormalize(values64)
    stability = _max_abs(torch_backend.qnorm(output.detach().cpu().to(torch.float64)) - 1.0)
    scalar_error = _scalar_check_qnormalize(output, values64, dtype_name)
    return _result_from_output(
        suite,
        case,
        backend,
        device,
        dtype,
        dtype_name,
        output,
        reference,
        elapsed_s,
        stability_max_abs=stability,
        scalar_reference_max_abs_error=scalar_error,
        execution_label=execution_label,
        stable_benchmark=stable_benchmark,
        methodology_note=methodology_note,
    )


def _run_qinverse(
    suite: str,
    case: BenchmarkCase,
    backend: str,
    device: torch.device,
    dtype: torch.dtype,
    dtype_name: str,
    generator: torch.Generator,
    execution_label: ExecutionLabel,
    stable_benchmark: bool,
    methodology_note: str,
) -> BenchmarkResult:
    values64 = _scaled_quaternions(case.items, generator, min_scale=1e-1, max_scale=1e1)
    values = _to_device_dtype(values64, device, dtype)

    output, elapsed_s = _measure(
        lambda: torch_backend.qinverse(values),
        case.warmup,
        case.iterations,
        device,
    )
    reference = torch_backend.qinverse(values64)
    residual = torch_backend.qmul(values, output).detach().cpu().to(torch.float64)
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64).expand_as(residual)
    stability = _max_abs(residual - identity)
    scalar_error = _scalar_check_qinverse(output, values64, dtype_name)
    return _result_from_output(
        suite,
        case,
        backend,
        device,
        dtype,
        dtype_name,
        output,
        reference,
        elapsed_s,
        stability_max_abs=stability,
        scalar_reference_max_abs_error=scalar_error,
        execution_label=execution_label,
        stable_benchmark=stable_benchmark,
        methodology_note=methodology_note,
    )


def _run_phase_update(
    suite: str,
    case: BenchmarkCase,
    backend: str,
    device: torch.device,
    dtype: torch.dtype,
    dtype_name: str,
    generator: torch.Generator,
    execution_label: ExecutionLabel,
    stable_benchmark: bool,
    methodology_note: str,
) -> BenchmarkResult:
    phase64 = _rand((case.items,), generator) * (2.0 * math.pi) - math.pi
    rate64 = _randn((case.items,), generator)
    amplitude64 = _rand((case.items,), generator)
    phase = _to_device_dtype(phase64, device, dtype)
    rate = _to_device_dtype(rate64, device, dtype)
    amplitude = _to_device_dtype(amplitude64, device, dtype)
    dt = 0.01

    def op() -> torch.Tensor:
        return torch_backend.phase_update(phase, rate, amplitude, dt)

    output, elapsed_s = _measure(op, case.warmup, case.iterations, device)
    reference = torch_backend.phase_update(phase64, rate64, amplitude64, dt)
    return _result_from_output(
        suite,
        case,
        backend,
        device,
        dtype,
        dtype_name,
        output,
        reference,
        elapsed_s,
        scalar_reference_max_abs_error=None,
        execution_label=execution_label,
        stable_benchmark=stable_benchmark,
        methodology_note=methodology_note,
    )


def _result_from_output(
    suite: str,
    case: BenchmarkCase,
    backend: str,
    device: torch.device | str,
    dtype: torch.dtype,
    dtype_name: str,
    output: torch.Tensor,
    reference: torch.Tensor,
    elapsed_s: float,
    *,
    stability_max_abs: float | None = None,
    scalar_reference_max_abs_error: float | None,
    execution_label: ExecutionLabel | None = None,
    stable_benchmark: bool = False,
    methodology_note: str | None = None,
    correctness: dict[str, object] | None = None,
    timing: dict[str, object] | None = None,
    provenance: dict[str, object] | None = None,
    implementation_class: str | None = None,
    performance_eligible: bool = False,
) -> BenchmarkResult:
    if not torch.isfinite(output).all() or not torch.isfinite(reference).all():
        raise ValueError("benchmark output/reference contains non-finite values")
    errors = _error_metrics(output, reference)
    hardware = _hardware_estimate(case, dtype, elapsed_s)
    latency_ms = elapsed_s * 1000.0 / case.iterations
    throughput = case.items * case.iterations / elapsed_s
    checksum = float(output.detach().cpu().to(torch.float64).sum().item())
    return BenchmarkResult(
        suite=suite,
        workload=case.workload,
        backend=backend,
        device=str(device),
        dtype=dtype_name,
        items=case.items,
        iterations=case.iterations,
        warmup=case.warmup,
        structured_shape=_structured_shape(case),
        throughput_unit=case.throughput_unit,
        elapsed_s=elapsed_s,
        latency_ms=latency_ms,
        throughput=throughput,
        max_abs_error=errors["max_abs_error"],
        max_rel_error=errors["max_rel_error"],
        rms_error=errors["rms_error"],
        stability_max_abs=stability_max_abs,
        scalar_reference_max_abs_error=scalar_reference_max_abs_error,
        estimated_flops=hardware["estimated_flops"],
        estimated_flops_per_s=hardware["estimated_flops_per_s"],
        estimated_bytes_read=hardware["estimated_bytes_read"],
        estimated_bytes_written=hardware["estimated_bytes_written"],
        estimated_total_bytes=hardware["estimated_total_bytes"],
        effective_gb_per_s=hardware["effective_gb_per_s"],
        arithmetic_intensity_flops_per_byte=hardware[
            "arithmetic_intensity_flops_per_byte"
        ],
        checksum=checksum,
        execution_label=execution_label or _default_execution_label(backend),
        stable_benchmark=stable_benchmark,
        methodology_note=methodology_note
        or _methodology_note(
            backend,
            execution_label or _default_execution_label(backend),
            None,
        ),
        correctness=correctness
        or {
            "passed": True,
            "atol": SCALAR_ERROR_TOLERANCES[dtype_name],
            "rtol": SCALAR_ERROR_TOLERANCES[dtype_name],
            "failing_values": 0,
            "nonfinite_values": 0,
            "validated_values": output.numel(),
            "whole_output_max_abs_error": errors["max_abs_error"],
            "scalar_first_eight_max_abs_error": scalar_reference_max_abs_error,
            "golden": "backend reference plus independent scalar diagnostic",
        },
        timing=timing
        or {
            "repetitions": 1,
            "device_s": timing_summary([elapsed_s]),
            "primary_elapsed": "device_s.median",
        },
        provenance=provenance,
        implementation_class=implementation_class,
        performance_eligible=performance_eligible,
    )


def _measure(
    op: Callable[[], torch.Tensor],
    warmup: int,
    iterations: int,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    output: torch.Tensor | None = None
    with torch.no_grad():
        for _ in range(warmup):
            output = op()
        _sync(device)
        start = time.perf_counter()
        for _ in range(iterations):
            output = op()
        _sync(device)
        elapsed_s = time.perf_counter() - start

    if output is None:
        output = op()
    return output, elapsed_s


def _run_external_command(
    external_command: str,
    *,
    work_dir: Path,
    manifest_path: Path,
    execution_label: ExecutionLabel | None = None,
) -> float:
    command = shlex.split(external_command)
    if not command:
        raise ValueError("--external-command must not be empty")

    env = os.environ.copy()
    env["TT_RQM_EXTERNAL_QMUL_DIR"] = str(work_dir)
    env["TT_RQM_EXTERNAL_QMUL_MANIFEST"] = str(manifest_path)
    if execution_label is not None:
        env["TT_RQM_EXECUTION_LABEL"] = execution_label
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=env,
    )
    elapsed_s = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            "external-qmul command failed\n"
            f"command: {' '.join(command)}\n"
            f"work_dir: {work_dir}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return elapsed_s


def _load_external_metrics(path: Path) -> dict[str, object]:
    if not path.exists():
        raise RuntimeError(f"external-qmul command did not write {path.name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("external-qmul metrics.json must contain a JSON object")
    return payload


def _write_float32_binary(path: Path, value: torch.Tensor) -> None:
    flat = value.detach().cpu().to(dtype=torch.float32).contiguous().reshape(-1)
    payload = array("f", flat.tolist())
    if sys.byteorder != "little":
        payload.byteswap()
    path.write_bytes(payload.tobytes())


def _read_float32_binary(path: Path, shape: tuple[int, ...]) -> torch.Tensor:
    if not path.exists():
        raise RuntimeError(f"external-qmul command did not write {path.name}")

    expected_values = math.prod(shape)
    expected_bytes = expected_values * torch.empty((), dtype=torch.float32).element_size()
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"external-qmul output size mismatch for {path.name}: "
            f"expected {expected_bytes} bytes, got {actual_bytes}"
        )

    payload = array("f")
    payload.frombytes(path.read_bytes())
    if sys.byteorder != "little":
        payload.byteswap()
    return torch.tensor(payload, dtype=torch.float32).reshape(shape)


def _error_metrics(output: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    output64 = output.detach().cpu().to(torch.float64)
    reference64 = reference.detach().cpu().to(torch.float64)
    diff = output64 - reference64
    abs_diff = torch.abs(diff)
    denom = torch.clamp(torch.abs(reference64), min=1e-12)
    rel_diff = abs_diff / denom
    return {
        "max_abs_error": _max_abs(diff),
        "max_rel_error": float(torch.max(rel_diff).item()),
        "rms_error": float(torch.sqrt(torch.mean(diff * diff)).item()),
    }


def _hardware_estimate(
    case: BenchmarkCase,
    dtype: torch.dtype,
    elapsed_s: float,
) -> dict[str, float | int]:
    bytes_per_value = torch.empty((), dtype=dtype).element_size()
    flops_per_item, read_values_per_item, written_values_per_item = _per_item_estimate(
        case.workload
    )
    measured_items = case.items * case.iterations
    estimated_flops = int(flops_per_item * measured_items)
    estimated_bytes_read = int(read_values_per_item * bytes_per_value * measured_items)
    estimated_bytes_written = int(
        written_values_per_item * bytes_per_value * measured_items
    )
    estimated_total_bytes = estimated_bytes_read + estimated_bytes_written
    return {
        "estimated_flops": estimated_flops,
        "estimated_flops_per_s": estimated_flops / elapsed_s,
        "estimated_bytes_read": estimated_bytes_read,
        "estimated_bytes_written": estimated_bytes_written,
        "estimated_total_bytes": estimated_total_bytes,
        "effective_gb_per_s": estimated_total_bytes / elapsed_s / 1e9,
        "arithmetic_intensity_flops_per_byte": (
            estimated_flops / estimated_total_bytes if estimated_total_bytes else 0.0
        ),
    }


def _per_item_estimate(workload: str) -> tuple[int, int, int]:
    if workload == "qmul":
        # 28 FLOPs: 16 multiplies plus 12 additions/subtractions.
        # Logical memory traffic: two 4-lane quaternion reads, one 4-lane write.
        return 28, 8, 4
    if workload == "qrotate":
        # Conservative estimate: two Hamilton products (56 FLOPs) plus 8 FLOPs
        # for conjugate/vector-packing overhead. Logical traffic counts one
        # rotor read, one vector read, and one rotated vector write.
        return 64, 7, 3
    if workload == "qnormalize":
        # Norm estimate: 4 squares, 3 adds, 1 sqrt, 1 reciprocal, 4 output
        # scaling/division operations.
        return 13, 4, 4
    if workload == "qinverse":
        # Norm-squared plus conjugate and reciprocal scaling: 4 squares, 3 adds,
        # 3 sign flips, 1 reciprocal, and 4 output scaling/division operations.
        return 15, 4, 4
    if workload == "phase_update":
        # Transcendental-heavy: phase multiply/add plus sin/cos generation and
        # amplitude scaling. Each transcendental is counted as one reported op.
        return 6, 3, 2
    raise ValueError(f"unsupported workload estimate: {workload}")


def _scalar_check_qmul(
    output: torch.Tensor,
    a64: torch.Tensor,
    b64: torch.Tensor,
    dtype_name: str,
) -> float:
    sample_count = min(SCALAR_CHECK_SAMPLES, output.shape[0])
    expected = [
        scalar_reference.qmul_scalar(a64[index].tolist(), b64[index].tolist())
        for index in range(sample_count)
    ]
    error = _sample_error(output, expected, sample_count)
    _validate_scalar_error("qmul", error, dtype_name)
    return error


def _scalar_check_qrotate(
    output: torch.Tensor,
    rotors64: torch.Tensor,
    vectors64: torch.Tensor,
    dtype_name: str,
) -> float:
    sample_count = min(SCALAR_CHECK_SAMPLES, output.shape[0])
    expected = [
        scalar_reference.qrotate_vector_scalar(
            rotors64[index].tolist(),
            vectors64[index].tolist(),
        )
        for index in range(sample_count)
    ]
    error = _sample_error(output, expected, sample_count)
    _validate_scalar_error("qrotate", error, dtype_name)
    return error


def _scalar_check_qnormalize(
    output: torch.Tensor,
    values64: torch.Tensor,
    dtype_name: str,
) -> float:
    sample_count = min(SCALAR_CHECK_SAMPLES, output.shape[0])
    expected = [
        scalar_reference.qnormalize_scalar(values64[index].tolist())
        for index in range(sample_count)
    ]
    error = _sample_error(output, expected, sample_count)
    _validate_scalar_error("qnormalize", error, dtype_name)
    return error


def _scalar_check_qinverse(
    output: torch.Tensor,
    values64: torch.Tensor,
    dtype_name: str,
) -> float:
    sample_count = min(SCALAR_CHECK_SAMPLES, output.shape[0])
    expected = [
        scalar_reference.qinverse_scalar(values64[index].tolist())
        for index in range(sample_count)
    ]
    error = _sample_error(output, expected, sample_count)
    _validate_scalar_error("qinverse", error, dtype_name)
    return error


def _sample_error(
    output: torch.Tensor,
    expected: list[tuple[float, ...]],
    sample_count: int,
) -> float:
    expected_tensor = torch.tensor(expected, dtype=torch.float64)
    actual_tensor = output[:sample_count].detach().cpu().to(torch.float64)
    return _max_abs(actual_tensor - expected_tensor)


def _validate_scalar_error(workload: str, error: float, dtype_name: str) -> None:
    tolerance = SCALAR_ERROR_TOLERANCES[dtype_name]
    if error > tolerance:
        raise ValueError(
            f"{workload} scalar reference check failed: "
            f"max_abs_error={error:g}, tolerance={tolerance:g}"
        )


def _randn(shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    return torch.randn(shape, generator=generator, dtype=torch.float64)


def _rand(shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    return torch.rand(shape, generator=generator, dtype=torch.float64)


def _scaled_quaternions(
    items: int,
    generator: torch.Generator,
    *,
    min_scale: float,
    max_scale: float,
) -> torch.Tensor:
    values = _randn((items, 4), generator)
    scales = torch.logspace(
        math.log10(min_scale),
        math.log10(max_scale),
        steps=items,
        dtype=torch.float64,
    ).unsqueeze(-1)
    return values * scales


def _to_device_dtype(
    value: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return value.to(device=device, dtype=dtype)


def _max_abs(value: torch.Tensor) -> float:
    return float(torch.max(torch.abs(value)).item())


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _structured_shape(case: BenchmarkCase) -> str:
    if case.workload in {"qmul", "qnormalize", "qinverse"}:
        return f"[{case.items}, 4]"
    if case.workload == "qrotate":
        return f"rotor=[{case.items}, 4], vector=[{case.items}, 3]"
    if case.workload == "phase_update":
        return f"phase=[{case.items}], state=[{case.items}, 2]"
    return f"[{case.items}]"


def _format_row(values: list[str], widths: list[int]) -> str:
    return "  ".join(value.ljust(width) for value, width in zip(values, widths, strict=True))


def _report_results(report: dict[str, object]) -> list[dict[str, object]]:
    results = report["results"]
    if not isinstance(results, list):
        raise TypeError("report results must be a list")
    for result in results:
        if not isinstance(result, dict):
            raise TypeError("each result must be a dict")
    return results


def _optional_scientific(value: object) -> str:
    return "-" if value is None else f"{float(value):.3e}"


def _backend_note(report: dict[str, object]) -> str:
    if report.get("backend") == "tt-lang-sim":
        return (
            "- Current results use the TT-Lang functional simulator. They "
            "validate kernel logic and report shape, not hardware performance."
        )
    if report.get("backend") == "external-qmul":
        return (
            "- Current results use the external-qmul candidate harness. "
            "StructuredBench validates the whole output against an independent "
            "float64 golden calculation; hardware claims depend on the external command "
            "and measurement environment."
        )
    return "- Current results use the CPU/PyTorch reference backend."


def _report_intro(report: dict[str, object]) -> list[str]:
    execution_label = report.get("execution_label")
    results = report.get("results", [])
    legacy = isinstance(results, list) and any(
        isinstance(result, dict) and "correctness" not in result for result in results
    )
    legacy_note = (
        "This is a historical pre-integrity artifact and has not been rerun under "
        "the whole-output/metrics-v2 gate."
        if legacy
        else ""
    )
    if report.get("backend") == "tt-lang-sim":
        return [
            (
                "This report demonstrates that the `[N, 4]` `qmul` contract can "
                "be exercised through the TT-Lang functional simulator and "
                "validated through the current conformance contract. It is a "
                "logic and report-shape artifact, not hardware performance evidence."
            ),
            "",
            *([legacy_note, ""] if legacy_note else []),
            "Next evidence target: `reports/tt_emule_qmul_candidate.md`.",
            "Final target: `reports/tt_hardware_qmul_quickstart.md`.",
            "",
        ]
    if report.get("backend") == "external-qmul" and execution_label == "emulation":
        return [
            (
                "This report demonstrates the `external-qmul` candidate protocol "
                "with an emulation-labeled run. StructuredBench validates the "
                "candidate output through the current conformance contract. It "
                "is not hardware performance evidence."
            ),
            "",
            *([legacy_note, ""] if legacy_note else []),
            "Final target: `reports/tt_hardware_qmul_quickstart.md`.",
            "",
        ]
    return []


def _methodology_note_line(report: dict[str, object]) -> str:
    note = report.get("methodology_note")
    if not note:
        return "- Methodology note: not provided."
    return f"- Methodology note: {note}"


def _committed_report_note(report: dict[str, object]) -> str:
    if report.get("backend") == "tt-lang-sim":
        return (
            "- This committed TT-Lang report is a simulator smoke output. It is "
            "included to show the report shape, not to claim stable hardware performance."
        )
    if report.get("backend") == "external-qmul":
        return (
            "- External-qmul reports are candidate-command outputs validated by "
            "StructuredBench. They should not be read as Tenstorrent hardware "
            "performance unless the command and device are explicitly documented."
        )
    return (
        "- Committed reports are sample CPU/PyTorch reference outputs. They are "
        "included to show the report shape and outreach packet format, not to "
        "claim stable hardware performance."
    )


def _backend_metadata_notes(report: dict[str, object]) -> list[str]:
    if report.get("backend") != "tt-lang-sim":
        return []
    metadata = report.get("tt_lang_sim", {})
    if not isinstance(metadata, dict):
        return []
    fields = [
        f"seed={report.get('seed')}",
        f"layout={metadata.get('layout')}",
        f"block_items={metadata.get('block_items')}",
        f"padded_items={metadata.get('padded_items')}",
        f"variant={metadata.get('variant')}",
        f"sim_cli={metadata.get('sim_cli')}",
        f"sim_version={metadata.get('sim_version')}",
        f"stats_cli={metadata.get('stats_cli')}",
        f"trace_enabled={metadata.get('trace_enabled', False)}",
    ]
    notes = ["- Simulator metadata: " + ", ".join(fields) + "."]
    if metadata.get("variant_note"):
        notes.append(f"- TT-Lang variant note: {metadata.get('variant_note')}.")
    return notes


def _tt_lang_trace_stats_section(report: dict[str, object]) -> list[str]:
    if report.get("backend") != "tt-lang-sim":
        return []
    metadata = report.get("tt_lang_sim", {})
    if not isinstance(metadata, dict) or not metadata.get("trace_enabled"):
        return []

    stats_available = "yes" if metadata.get("stats_available") else "no"
    lines = [
        "## TT-Lang Simulator Trace/Stats",
        "",
        "- Trace capture: enabled.",
        f"- Trace path: {_format_trace_path(metadata.get('trace_path'))}.",
        f"- Stats CLI available: {stats_available}.",
        "- These trace/stat outputs are simulator diagnostics, not hardware performance.",
        "",
    ]

    stats_summary = metadata.get("stats_summary")
    if stats_summary:
        lines.extend(
            [
                "```text",
                _clip_markdown_block(str(stats_summary)),
                "```",
                "",
            ]
        )
        return lines

    stats_error = metadata.get("stats_error")
    if stats_error:
        lines.extend(
            [
                "Stats summary unavailable:",
                "",
                "```text",
                _clip_markdown_block(str(stats_error)),
                "```",
                "",
            ]
        )
    return lines


def _format_trace_path(value: object) -> str:
    if value is None:
        return "temporary trace file, not retained"
    if isinstance(value, list):
        if not value:
            return "temporary trace file, not retained"
        return ", ".join(f"`{item}`" for item in value)
    return f"`{value}`"


def _clip_markdown_block(value: str, *, limit: int = 4000) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 32] + "\n... truncated for Markdown report ..."


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
