#!/usr/bin/env python3
"""Build the experimental TT-Metalium qmul candidate when SDK deps exist."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

from check_environment import main as check_environment_main

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_BUILD_DIR = PACKAGE_DIR / "build"
DEFAULT_BINARY_NAME = "tt_rqm_metalium_qmul_candidate"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the experimental TT-Metalium qmul candidate."
    )
    parser.add_argument(
        "--tt-metal-root",
        type=Path,
        default=None,
        help="Path to a tt-metal / TT-Metalium checkout.",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=DEFAULT_BUILD_DIR,
        help="CMake build directory for the candidate package.",
    )
    parser.add_argument(
        "--cmake-prefix-path",
        type=Path,
        default=None,
        help=(
            "Built/installed TT-Metalium prefix. Defaults to common build "
            "directories under --tt-metal-root, such as build_emule."
        ),
    )
    parser.add_argument(
        "--generator",
        default=None,
        help="Optional CMake generator. Ninja is used when available.",
    )
    args = parser.parse_args(argv)

    check_args = []
    if args.tt_metal_root is not None:
        check_args.extend(["--tt-metal-root", str(args.tt_metal_root)])
    check_status = check_environment_main(check_args)
    if check_status != 0:
        print(
            "Build stopped before configuring the candidate. A TT-Metalium SDK "
            "checkout is required.",
            file=sys.stderr,
        )
        return check_status

    cmake = shutil.which("cmake")
    if cmake is None:
        print(
            "Build unavailable: cmake was not found on PATH. Use an x86-64 "
            "Linux TT-Metalium build environment with CMake, Ninja, and "
            "clang-20 installed.",
            file=sys.stderr,
        )
        return 2

    tt_metal_root = _resolve_tt_metal_root(args.tt_metal_root)
    prefix = args.cmake_prefix_path or _default_prefix(tt_metal_root)
    if prefix is None:
        print(
            "Build unavailable: no built TT-Metalium CMake package was found. "
            "Build tt-metal first, for example with build_emule and "
            "TT_METAL_USE_EMULE=ON, then pass --cmake-prefix-path.",
            file=sys.stderr,
        )
        return 2

    configure = [
        cmake,
        "-S",
        str(PACKAGE_DIR),
        "-B",
        str(args.build_dir),
        f"-DCMAKE_PREFIX_PATH={prefix}",
    ]
    generator = args.generator or ("Ninja" if shutil.which("ninja") else None)
    if generator is not None:
        configure.extend(["-G", generator])

    env = os.environ.copy()
    if tt_metal_root is not None:
        env.setdefault("TT_METAL_HOME", str(tt_metal_root))
        env.setdefault("TT_METAL_RUNTIME_ROOT", str(tt_metal_root))

    try:
        subprocess.run(configure, cwd=PACKAGE_DIR, check=True, env=env)
        subprocess.run(
            [cmake, "--build", str(args.build_dir), "--target", DEFAULT_BINARY_NAME],
            cwd=PACKAGE_DIR,
            check=True,
            env=env,
        )
    except subprocess.CalledProcessError as exc:
        print(
            "Build failed while compiling the experimental TT-Metalium qmul "
            "candidate. This is not a hardware performance result.",
            file=sys.stderr,
        )
        return exc.returncode or 2

    binary = args.build_dir / DEFAULT_BINARY_NAME
    print(f"Built candidate: {binary}")
    print("Validate with scripts/validate_qmul_candidate.py before reporting results.")
    return 0


def _resolve_tt_metal_root(cli_root: Path | None) -> Path | None:
    if cli_root is not None:
        return cli_root.expanduser().resolve()
    for name in ("TT_METAL_HOME", "TT_METALIUM_HOME"):
        value = os.environ.get(name)
        if value:
            return Path(value).expanduser().resolve()
    return None


def _default_prefix(tt_metal_root: Path | None) -> Path | None:
    candidates: list[Path] = []
    env_prefix = os.environ.get("TT_METALIUM_PREFIX") or os.environ.get(
        "TT_METAL_INSTALL_PREFIX"
    )
    if env_prefix:
        candidates.append(Path(env_prefix).expanduser())
    if tt_metal_root is not None:
        candidates.extend(
            [
                tt_metal_root / "build_emule",
                tt_metal_root / "build",
                tt_metal_root / "build_Release",
                tt_metal_root / "build_RelWithDebInfo",
                tt_metal_root / ".build" / "default",
            ]
        )
    for candidate in candidates:
        if _has_metalium_package(candidate):
            return candidate.resolve()
    return None


def _has_metalium_package(prefix: Path) -> bool:
    return any(
        path.exists()
        for path in (
            prefix / "lib" / "cmake" / "tt-metalium" / "TT-MetaliumConfig.cmake",
            prefix / "lib64" / "cmake" / "tt-metalium" / "TT-MetaliumConfig.cmake",
            prefix / "tt_metal" / "cmake" / "TT-MetaliumConfig.cmake",
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
