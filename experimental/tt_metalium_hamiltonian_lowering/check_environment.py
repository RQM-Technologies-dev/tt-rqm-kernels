#!/usr/bin/env python3
"""Fail closed unless the H2A candidate uses the pinned TT-Metal checkout."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

PINNED_TT_METAL_COMMIT = "dd2849b5bc6b7a5d38a9eafbeba31ef8d530f8d4"


def validate_tt_metal_root(root: Path) -> tuple[str, bool]:
    root = root.expanduser().resolve()
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != PINNED_TT_METAL_COMMIT:
        raise ValueError(
            f"TT-Metal commit mismatch: expected {PINNED_TT_METAL_COMMIT}, got {commit}"
        )
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return commit, not dirty


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tt-metal-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        commit, clean = validate_tt_metal_root(args.tt_metal_root)
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        print(exc)
        return 2
    print(f"TT-Metal commit: {commit}")
    print(f"TT-Metal tree clean: {str(clean).lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
