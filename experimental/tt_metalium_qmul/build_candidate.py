#!/usr/bin/env python3
"""Placeholder build command for a future TT-Metalium qmul candidate."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from check_environment import main as check_environment_main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare a future TT-Metalium qmul candidate build."
    )
    parser.add_argument(
        "--tt-metal-root",
        type=Path,
        default=None,
        help="Path to a tt-metal / TT-Metalium checkout.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/tt_metalium_qmul_candidate"),
        help="Future candidate binary path. No binary is emitted by this scaffold.",
    )
    args = parser.parse_args(argv)

    check_args = []
    if args.tt_metal_root is not None:
        check_args.extend(["--tt-metal-root", str(args.tt_metal_root)])
    check_status = check_environment_main(check_args)
    if check_status != 0:
        print(
            "Build stopped before emitting a candidate binary. This scaffold "
            "requires a real TT-Metalium SDK checkout before implementation.",
            file=sys.stderr,
        )
        return check_status

    print(
        "TT-Metalium SDK root detected, but no TT-Metalium qmul source is "
        "present in this scaffold yet. No binary was emitted and no hardware "
        "performance is claimed.",
        file=sys.stderr,
    )
    print(f"Future output path: {args.output}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
