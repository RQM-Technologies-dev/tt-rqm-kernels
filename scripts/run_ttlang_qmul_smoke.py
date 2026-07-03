#!/usr/bin/env python3
"""Run the optional TT-Lang simulator qmul smoke benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tt_rqm_kernels.backends.tt_lang import check_tt_lang_sim
from tt_rqm_kernels.backends.tt_lang.availability import TTLangSimulatorUnavailable
from tt_rqm_kernels.backends.tt_lang.runner import SUPPORTED_VARIANTS, run_qmul_cases
from tt_rqm_kernels.structuredbench import BenchmarkCase, render_markdown_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run TT-Lang simulator qmul and emit a StructuredBench report."
    )
    parser.add_argument("--items", type=_positive_int, default=128)
    parser.add_argument("--iters", type=_positive_int, default=1)
    parser.add_argument("--warmup", type=_nonnegative_int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-output", type=Path, default=Path("reports/tt_lang_qmul_sim.json"))
    parser.add_argument("--markdown-output", type=Path, default=Path("reports/tt_lang_qmul_sim.md"))
    parser.add_argument("--sim-cli", default=None)
    parser.add_argument(
        "--variant",
        choices=SUPPORTED_VARIANTS,
        default="block",
        help="TT-Lang qmul simulator variant to run.",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help=(
            "Enable TT-Lang simulator trace capture. Uses a temporary trace file "
            "unless --trace-output is provided."
        ),
    )
    parser.add_argument(
        "--trace-output",
        type=Path,
        default=None,
        help="Write the TT-Lang simulator JSONL trace to this path.",
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=None,
        help=(
            "Write tt-lang-sim-stats text output when trace capture is enabled. "
            "Also enables trace capture."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check simulator availability without running the benchmark.",
    )
    args = parser.parse_args(argv)

    if args.check:
        availability = check_tt_lang_sim(sim_cli=args.sim_cli)
        print(
            json.dumps(
                {
                    "available": availability.available,
                    "sim_cli": availability.sim_cli,
                    "version": availability.version,
                    "stats_cli": availability.stats_cli,
                    "stats_available": availability.stats_available,
                    "stats_reason": availability.stats_reason,
                    "reason": availability.reason,
                    "setup_hint": availability.setup_hint,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    case = BenchmarkCase(
        workload="qmul",
        items=args.items,
        iterations=args.iters,
        warmup=args.warmup,
        throughput_unit="qmul/s",
    )
    try:
        report = run_qmul_cases(
            [case],
            seed=args.seed,
            sim_cli=args.sim_cli,
            variant=args.variant,
            trace=args.trace,
            trace_output=args.trace_output,
            stats_output=args.stats_output,
        )
    except TTLangSimulatorUnavailable as exc:
        print(str(exc), file=sys.stderr)
        return 2

    _write_text(args.json_output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_text(args.markdown_output, render_markdown_report(report))
    print(json.dumps(report, indent=2, sort_keys=True))
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
