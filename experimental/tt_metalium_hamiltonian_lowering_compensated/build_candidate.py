#!/usr/bin/env python3
"""Build the compensated single-core H2A candidate variant."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess

PACKAGE = Path(__file__).resolve().parent
TARGET = "tt_rqm_metalium_hamiltonian_lowering_compensated_candidate"
DIAGNOSTIC_TARGETS = (
    "tt_rqm_metalium_hamiltonian_lowering_compensated_product_diagnostic",
    "tt_rqm_metalium_hamiltonian_lowering_compensated_value_diagnostic",
    "tt_rqm_metalium_hamiltonian_lowering_compensated_trig_diagnostic",
    "tt_rqm_metalium_hamiltonian_lowering_original_trig_diagnostic",
)


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
    prefixes = [args.cmake_prefix_path] if args.cmake_prefix_path else [
        root / "build_Release",
        root / "build",
        root / "build_RelWithDebInfo",
        root / ".build" / "default",
    ]
    prefix = next(
        (path.resolve() for path in prefixes if path and _package_dir(path.resolve())), None
    )
    if prefix is None:
        raise SystemExit("No built TT-Metalium CMake package was found")
    package_dir = _package_dir(prefix)
    assert package_dir is not None
    build = args.build_dir.expanduser().resolve()
    cmake = shutil.which("cmake")
    if cmake is None:
        raise SystemExit("cmake is required")
    env = os.environ.copy()
    env["TT_METAL_HOME"] = str(root)
    env["TT_METAL_RUNTIME_ROOT"] = str(root)
    ulfm = Path("/opt/openmpi-v5.0.7-ulfm")
    if ulfm.exists():
        env["PATH"] = f"{ulfm / 'bin'}:{env.get('PATH', '')}"
        env["LD_LIBRARY_PATH"] = f"{ulfm / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}"
    command = [
        cmake,
        "-S",
        str(PACKAGE),
        "-B",
        str(build),
        "-G",
        "Ninja",
        f"-DCMAKE_PREFIX_PATH={prefix}",
        f"-DTT-Metalium_DIR={package_dir}",
    ]
    if (ulfm / "bin" / "mpicxx").exists():
        command.append(f"-DMPI_CXX_COMPILER={ulfm / 'bin' / 'mpicxx'}")
    subprocess.run(command, check=True, env=env)
    subprocess.run(
        [cmake, "--build", str(build), "--target", TARGET, *DIAGNOSTIC_TARGETS],
        check=True,
        env=env,
    )
    print(build / TARGET)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
