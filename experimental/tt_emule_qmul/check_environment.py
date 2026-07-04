#!/usr/bin/env python3
"""Check whether a tt-emule development environment is available for qmul."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import sys


TT_METAL_ENV_VARS = ("TT_METAL_HOME", "TT_METALIUM_HOME")
TT_EMULE_ENV_VARS = ("TT_EMULE_HOME",)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check for a local tt-metal + tt-emule checkout."
    )
    parser.add_argument(
        "--tt-metal-root",
        type=Path,
        default=None,
        help="Path to a tt-metal / TT-Metalium checkout.",
    )
    parser.add_argument(
        "--tt-emule-root",
        type=Path,
        default=None,
        help="Path to a tt-emule checkout.",
    )
    parser.add_argument(
        "--skip-platform-check",
        action="store_true",
        help="Skip the x86-64 Linux check. Intended for tests only.",
    )
    args = parser.parse_args(argv)

    if not args.skip_platform_check and not _is_x86_64_linux():
        print(
            "tt-emule environment unavailable: tt-emule is expected to run on "
            "x86-64 Linux. Current platform is "
            f"{platform.system()} {platform.machine()}.",
            file=sys.stderr,
        )
        return 2

    tt_metal_root = args.tt_metal_root or _root_from_env(TT_METAL_ENV_VARS)
    tt_emule_root = args.tt_emule_root or _root_from_env(TT_EMULE_ENV_VARS)

    tt_metal_error = _validate_tt_metal_root(tt_metal_root)
    if tt_metal_error is not None:
        print(f"tt-emule environment unavailable: {tt_metal_error}", file=sys.stderr)
        return 2

    tt_emule_error = _validate_tt_emule_root(tt_emule_root)
    if tt_emule_error is not None:
        print(f"tt-emule environment unavailable: {tt_emule_error}", file=sys.stderr)
        return 2

    assert tt_metal_root is not None
    assert tt_emule_root is not None
    print(f"tt-metal root detected: {tt_metal_root.expanduser().resolve()}")
    print(f"tt-emule root detected: {tt_emule_root.expanduser().resolve()}")
    print("tt-emule qmul preflight passed. This does not run a kernel.")
    return 0


def _is_x86_64_linux() -> bool:
    return platform.system() == "Linux" and platform.machine() in {"x86_64", "AMD64"}


def _root_from_env(names: tuple[str, ...]) -> Path | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return Path(value)
    return None


def _validate_tt_metal_root(root: Path | None) -> str | None:
    if root is None:
        return "set TT_METAL_HOME or TT_METALIUM_HOME to a local tt-metal checkout."
    root = root.expanduser().resolve()
    if not root.exists():
        return f"tt-metal checkout does not exist: {root}"
    if not root.is_dir():
        return f"tt-metal checkout is not a directory: {root}"
    markers = [root / "CMakeLists.txt", root / "tt_metal", root / "ttnn"]
    if not any(marker.exists() for marker in markers):
        return f"{root} does not look like a tt-metal checkout."
    return None


def _validate_tt_emule_root(root: Path | None) -> str | None:
    if root is None:
        return "set TT_EMULE_HOME to a local tt-emule checkout."
    root = root.expanduser().resolve()
    if not root.exists():
        return f"tt-emule checkout does not exist: {root}"
    if not root.is_dir():
        return f"tt-emule checkout is not a directory: {root}"
    markers = [root / "CMakeLists.txt", root / "include" / "tt_emule"]
    if not any(marker.exists() for marker in markers):
        return f"{root} does not look like a tt-emule checkout."
    return None


if __name__ == "__main__":
    raise SystemExit(main())
