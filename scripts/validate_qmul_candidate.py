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
        report = run_suite(
            "qmul",
            backend="external-qmul",
            dtype_name="float32",
            seed=args.seed,
            items_override=args.items,
            iterations_override=args.iters,
            warmup_override=args.warmup,
            external_command=args.command,
            execution_label=args.execution_label,
            stable_benchmark=args.stable_benchmark,
            methodology_note=args.methodology_note,
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
