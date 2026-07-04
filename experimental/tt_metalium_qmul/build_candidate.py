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
            "Build unavailable: no usable built/installed TT-Metalium CMake "
            "package export was found. Build/install tt-metal first, for "
            "example with build_emule and TT_METAL_USE_EMULE=ON, then pass "
            "--cmake-prefix-path.",
            file=sys.stderr,
        )
        return 2
    metalium_dir = _metalium_package_dir(prefix)
    if metalium_dir is None:
        print(
            "Build unavailable: no usable TT-Metalium CMake package directory "
            f"was found under {prefix}.",
            file=sys.stderr,
        )
        return 2

    configure = [
        cmake,
        "-S",
        str(PACKAGE_DIR),
        "-B",
        str(args.build_dir),
        f"-DCMAKE_PREFIX_PATH={';'.join(str(path) for path in _cmake_prefix_paths(prefix))}",
        f"-DTT-Metalium_DIR={metalium_dir}",
    ]
    module_paths = _cmake_module_paths(prefix)
    if module_paths:
        configure.append(f"-DCMAKE_MODULE_PATH={';'.join(str(path) for path in module_paths)}")
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
    return _metalium_package_dir(prefix) is not None


def _metalium_package_dir(prefix: Path) -> Path | None:
    for config in _metalium_config_paths(prefix):
        if _is_usable_metalium_config(config):
            return config.parent.expanduser().resolve()
    return None


def _is_usable_metalium_config(config: Path) -> bool:
    return config.exists() and (config.parent / "Metalium.cmake").exists()


def _cmake_prefix_paths(prefix: Path) -> list[Path]:
    candidates = [
        prefix,
        prefix / "_deps" / "fmt-build",
        prefix / "_deps" / "nlohmann_json-build",
        prefix / "_deps" / "spdlog-build",
        prefix / "_deps" / "tt-logger-build",
        prefix / "_deps" / "tt-logger-build" / "cmake",
        prefix / "_deps" / "enchantum-build",
        prefix / "_deps" / "reflect-build",
        prefix / "tt_metal" / "third_party" / "umd",
        prefix / "tt_metal" / "third_party" / "tracy",
    ]
    return _dedupe_existing(candidates)


def _cmake_module_paths(prefix: Path) -> list[Path]:
    return _dedupe_existing([prefix / "CPM_modules"])


def _metalium_config_paths(prefix: Path) -> tuple[Path, ...]:
    return (
        prefix / "lib" / "cmake" / "tt-metalium" / "tt-metalium-config.cmake",
        prefix / "lib64" / "cmake" / "tt-metalium" / "tt-metalium-config.cmake",
        prefix / "lib" / "cmake" / "tt-metalium" / "TT-MetaliumConfig.cmake",
        prefix / "lib64" / "cmake" / "tt-metalium" / "TT-MetaliumConfig.cmake",
        prefix / "tt_metal" / "cmake" / "TT-MetaliumConfig.cmake",
        prefix / "tt-metalium-config.cmake",
    )


def _dedupe_existing(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    existing: list[Path] = []
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved.exists() and resolved not in seen:
            existing.append(resolved)
            seen.add(resolved)
    return existing


if __name__ == "__main__":
    raise SystemExit(main())
