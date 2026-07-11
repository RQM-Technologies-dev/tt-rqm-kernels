#!/usr/bin/env python3
"""Validate a future TT-Metalium qmul executable through external-qmul."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from tt_rqm_kernels.structuredbench import EXECUTION_LABELS


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLACEHOLDER = (
    f"{sys.executable} "
    f"{REPO_ROOT / 'experimental' / 'tt_metalium_qmul' / 'run_candidate.py'}"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the StructuredBench external-qmul validator against a future "
            "TT-Metalium qmul candidate command."
        )
    )
    parser.add_argument(
        "--candidate-command",
        default=DEFAULT_PLACEHOLDER,
        help=(
            "Candidate command to validate. Defaults to the local placeholder, "
            "which fails clearly until a real TT-Metalium executable exists."
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
            "Execution environment label for the candidate. Use emulation for "
            "tt-emule runs and hardware only for real Tenstorrent hardware."
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
    parser.add_argument("--format", choices=("table", "json"), default="table")
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--markdown-output", type=Path, default=None)
    args = parser.parse_args(argv)

    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "validate_qmul_candidate.py"),
        "--command",
        args.candidate_command,
        "--items",
        str(args.items),
        "--iters",
        str(args.iters),
        "--warmup",
        str(args.warmup),
        "--seed",
        str(args.seed),
        "--repetitions",
        str(args.repetitions),
        "--format",
        args.format,
    ]
    if args.execution_label is not None:
        command.extend(["--execution-label", args.execution_label])
    if args.benchmark_stage is not None:
        command.extend(["--benchmark-stage", args.benchmark_stage])
    if args.stable_benchmark:
        command.append("--stable-benchmark")
    if args.methodology_note is not None:
        command.extend(["--methodology-note", args.methodology_note])
    if args.json_output is not None:
        command.extend(["--json-output", str(args.json_output)])
    if args.markdown_output is not None:
        command.extend(["--markdown-output", str(args.markdown_output)])

    completed = subprocess.run(command, cwd=REPO_ROOT)
    return completed.returncode


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


if __name__ == "__main__":
    raise SystemExit(main())
