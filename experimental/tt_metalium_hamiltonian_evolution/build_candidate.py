#!/usr/bin/env python3
"""Build the two-program H2B TT-Metalium candidate."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess

PACKAGE = Path(__file__).resolve().parent
TARGET = "tt_rqm_metalium_hamiltonian_evolution_candidate"


def _package_dir(prefix: Path) -> Path | None:
    for candidate in (
        prefix / "lib" / "cmake" / "tt-metalium",
        prefix / "lib64" / "cmake" / "tt-metalium",
        prefix / "tt_metal" / "cmake",
        prefix,
    ):
        if any(
            (candidate / name).exists()
            for name in ("tt-metalium-config.cmake", "TT-MetaliumConfig.cmake")
        ):
            return candidate
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tt-metal-root", type=Path, required=True)
    parser.add_argument("--cmake-prefix-path", type=Path)
    parser.add_argument("--build-dir", type=Path, default=PACKAGE / "build")
    args = parser.parse_args()
    root = args.tt_metal_root.expanduser().resolve()
    prefixes = (
        [args.cmake_prefix_path]
        if args.cmake_prefix_path
        else [root / "build_Release", root / "build", root / "build_RelWithDebInfo"]
    )
    prefix = next(
        (path.resolve() for path in prefixes if path and _package_dir(path.resolve())), None
    )
    if prefix is None:
        raise SystemExit("No built TT-Metalium CMake package was found")
    package_dir = _package_dir(prefix)
    assert package_dir is not None
    cmake = shutil.which("cmake")
    if cmake is None:
        raise SystemExit("cmake is required")
    build = args.build_dir.expanduser().resolve()
    env = os.environ.copy()
    env["TT_METAL_HOME"] = str(root)
    env["TT_METAL_RUNTIME_ROOT"] = str(root)
    generator = "Ninja" if shutil.which("ninja") else "Unix Makefiles"
    command = [
        cmake,
        "-S",
        str(PACKAGE),
        "-B",
        str(build),
        "-G",
        generator,
        f"-DCMAKE_PREFIX_PATH={prefix}",
        f"-DTT-Metalium_DIR={package_dir}",
    ]
    subprocess.run(command, check=True, env=env)
    subprocess.run([cmake, "--build", str(build), "--target", TARGET], check=True, env=env)
    print(build / TARGET)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
