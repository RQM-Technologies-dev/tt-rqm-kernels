#!/usr/bin/env python3
"""Validate a future TT-Metalium qmul executable through external-qmul."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


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
    parser.add_argument("--seed", type=int, default=0)
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
        "--format",
        args.format,
    ]
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
