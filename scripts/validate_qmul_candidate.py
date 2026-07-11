#!/usr/bin/env python3
"""Validate an external qmul candidate through StructuredBench."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from tt_rqm_kernels.structuredbench import (
    EXECUTION_LABELS,
    render_markdown_report,
    render_table,
    run_suite,
)
from tt_rqm_kernels.backends.tenstorrent.report import (
    validate_external_qmul_label,
    validate_stable_benchmark,
)

EXPECTED_HARDWARE_JSON_OUTPUT = Path("reports/tt_hardware_qmul_quickstart.json")
EXPECTED_HARDWARE_MARKDOWN_OUTPUT = Path("reports/tt_hardware_qmul_quickstart.md")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate an external qmul candidate with StructuredBench."
    )
    parser.add_argument(
        "--command",
        required=True,
        help=(
            "External candidate command. StructuredBench will expose "
            "TT_RQM_EXTERNAL_QMUL_DIR and TT_RQM_EXTERNAL_QMUL_MANIFEST."
        ),
    )
    parser.add_argument("--items", type=_positive_int, default=128)
    parser.add_argument("--iters", type=_positive_int, default=1)
    parser.add_argument("--warmup", type=_nonnegative_int, default=0)
    parser.add_argument("--repetitions", type=_positive_int, default=1)
    parser.add_argument(
        "--benchmark-stage",
        choices=("conformance", "performance"),
        default=None,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--execution-label",
        choices=EXECUTION_LABELS,
        default=None,
        help=(
            "Execution environment label for the external candidate. Use "
            "emulation or hardware only when the command really runs there."
        ),
    )
    parser.add_argument(
        "--stable-benchmark",
        action="store_true",
        help="Mark the candidate report as a stable benchmark.",
    )
    parser.add_argument(
        "--methodology-note",
        default=None,
        help="Optional short note describing the candidate measurement methodology.",
    )
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--markdown-output", type=Path, default=None)
    parser.add_argument("--format", choices=("table", "json"), default="table")
    args = parser.parse_args()

    try:
        _validate_report_args(
            command=args.command,
            execution_label=args.execution_label,
            stable_benchmark=args.stable_benchmark,
            methodology_note=args.methodology_note,
            json_output=args.json_output,
            markdown_output=args.markdown_output,
        )
        report = run_suite(
            "qmul",
            backend="external-qmul",
            dtype_name="float32",
            seed=args.seed,
            items_override=None if args.benchmark_stage == "performance" else args.items,
            iterations_override=None
            if args.benchmark_stage == "performance"
            else args.iters,
            warmup_override=None
            if args.benchmark_stage == "performance"
            else args.warmup,
            external_command=args.command,
            execution_label=args.execution_label,
            stable_benchmark=args.stable_benchmark,
            methodology_note=args.methodology_note,
            repetitions=args.repetitions,
            benchmark_stage=args.benchmark_stage,
        )
    except (RuntimeError, ValueError, TypeError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.json_output is not None:
        _write_text(args.json_output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.markdown_output is not None:
        _write_text(args.markdown_output, render_markdown_report(report))

    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_table(report))
    return 0


def _validate_report_args(
    *,
    command: str,
    execution_label: str | None,
    stable_benchmark: bool,
    methodology_note: str | None,
    json_output: Path | None,
    markdown_output: Path | None,
) -> None:
    if execution_label is None:
        return
    label = validate_external_qmul_label(execution_label, command=command)
    validate_stable_benchmark(label, stable_benchmark=stable_benchmark)
    if label != "hardware":
        return
    if not methodology_note or not methodology_note.strip():
        raise ValueError(
            "hardware-labeled external-qmul reports require --methodology-note "
            "describing the real Tenstorrent hardware environment; first "
            "samples should keep stable_benchmark=false."
        )
    _require_hardware_output_path(
        json_output,
        expected=EXPECTED_HARDWARE_JSON_OUTPUT,
        flag="--json-output",
    )
    _require_hardware_output_path(
        markdown_output,
        expected=EXPECTED_HARDWARE_MARKDOWN_OUTPUT,
        flag="--markdown-output",
    )


def _require_hardware_output_path(
    path: Path | None,
    *,
    expected: Path,
    flag: str,
) -> None:
    if path is None:
        raise ValueError(f"hardware-labeled reports require {flag} {expected}")
    if tuple(path.parts[-2:]) != tuple(expected.parts):
        raise ValueError(
            f"hardware-labeled reports require {flag} ending in {expected}"
        )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
