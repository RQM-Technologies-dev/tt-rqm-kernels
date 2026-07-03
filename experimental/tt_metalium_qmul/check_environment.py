#!/usr/bin/env python3
"""Check whether a TT-Metalium development environment is available."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


SDK_ENV_VARS = ("TT_METAL_HOME", "TT_METALIUM_HOME")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check for a local TT-Metalium SDK checkout."
    )
    parser.add_argument(
        "--tt-metal-root",
        type=Path,
        default=None,
        help="Path to a tt-metal / TT-Metalium checkout.",
    )
    args = parser.parse_args(argv)

    root = args.tt_metal_root or _root_from_env()
    if root is None:
        print(
            "TT-Metalium SDK unavailable: set TT_METAL_HOME or "
            "TT_METALIUM_HOME to a local tt-metal checkout.",
            file=sys.stderr,
        )
        return 2

    root = root.expanduser().resolve()
    if not root.exists():
        print(f"TT-Metalium SDK unavailable: {root} does not exist.", file=sys.stderr)
        return 2
    if not root.is_dir():
        print(f"TT-Metalium SDK unavailable: {root} is not a directory.", file=sys.stderr)
        return 2

    markers = [
        root / "tt_metal",
        root / "ttnn",
        root / "CMakeLists.txt",
    ]
    if not any(marker.exists() for marker in markers):
        print(
            "TT-Metalium SDK unavailable: "
            f"{root} does not look like a tt-metal checkout.",
            file=sys.stderr,
        )
        return 2

    print(f"TT-Metalium SDK candidate root detected: {root}")
    return 0


def _root_from_env() -> Path | None:
    for name in SDK_ENV_VARS:
        value = os.environ.get(name)
        if value:
            return Path(value)
    return None


if __name__ == "__main__":
    raise SystemExit(main())
