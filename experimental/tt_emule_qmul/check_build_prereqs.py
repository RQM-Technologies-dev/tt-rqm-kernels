#!/usr/bin/env python3
"""Check build prerequisites for the tt-emule qmul candidate path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import shutil
import subprocess
from typing import Any

from check_environment import (
    TT_EMULE_ENV_VARS,
    TT_METAL_ENV_VARS,
    _is_x86_64_linux,
    _root_from_env,
    _validate_tt_emule_root,
    _validate_tt_metal_root,
)

REQUIRED_TOOLS = ("git", "cmake", "ninja", "clang-20", "clang++-20")
REQUIRED_SUBMODULES = (
    "tt_metal/third_party/umd",
    "tt_metal/third_party/tracy",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check prerequisites for building qmul with tt-metal + tt-emule."
    )
    parser.add_argument("--tt-metal-root", type=Path, default=None)
    parser.add_argument("--tt-emule-root", type=Path, default=None)
    parser.add_argument(
        "--skip-platform-check",
        action="store_true",
        help="Skip the x86-64 Linux check. Intended for tests only.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON details.")
    args = parser.parse_args(argv)

    tt_metal_root = (args.tt_metal_root or _root_from_env(TT_METAL_ENV_VARS))
    tt_emule_root = (args.tt_emule_root or _root_from_env(TT_EMULE_ENV_VARS))
    checks: list[dict[str, Any]] = []

    if args.skip_platform_check or _is_x86_64_linux():
        checks.append(_ok("platform", f"{platform.system()} {platform.machine()}"))
    else:
        checks.append(
            _error(
                "platform",
                "tt-emule builds require x86-64 Linux; current platform is "
                f"{platform.system()} {platform.machine()}",
            )
        )

    tt_metal_error = _validate_tt_metal_root(tt_metal_root)
    if tt_metal_error is None and tt_metal_root is not None:
        tt_metal_root = tt_metal_root.expanduser().resolve()
        checks.append(_ok("tt-metal root", str(tt_metal_root)))
    else:
        checks.append(_error("tt-metal root", tt_metal_error or "missing"))

    tt_emule_error = _validate_tt_emule_root(tt_emule_root)
    if tt_emule_error is None and tt_emule_root is not None:
        tt_emule_root = tt_emule_root.expanduser().resolve()
        checks.append(_ok("tt-emule root", str(tt_emule_root)))
    else:
        checks.append(_error("tt-emule root", tt_emule_error or "missing"))

    for tool in REQUIRED_TOOLS:
        path = shutil.which(tool)
        checks.append(_ok(f"tool:{tool}", path) if path else _error(f"tool:{tool}", "not found on PATH"))

    if tt_metal_root is not None and tt_emule_root is not None and shutil.which("git"):
        checks.extend(_check_pin(tt_metal_root, tt_emule_root))
        checks.extend(_check_submodules(tt_metal_root))

    if tt_metal_root is not None:
        prefix = _find_metalium_package(tt_metal_root)
        if prefix is None:
            checks.append(
                _error(
                    "TT-Metalium CMake package",
                    "usable package export not found; build/install tt-metal with TT_METAL_USE_EMULE=ON before passing build_emule to build_candidate.py",
                )
            )
        else:
            checks.append(_ok("TT-Metalium CMake package", str(prefix)))

    payload = {
        "schema": "tt-rqm-tt-emule-build-prereqs.v1",
        "ok": all(check["status"] == "ok" for check in checks),
        "checks": checks,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("tt-emule qmul build prerequisites")
        for check in checks:
            print(f"{check['status']}: {check['name']}: {check['detail']}")
    return 0 if payload["ok"] else 2


def _check_pin(tt_metal_root: Path, tt_emule_root: Path) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    pin_path = tt_emule_root / "tt-metal-pin.txt"
    try:
        pin = _read_pin(pin_path)
    except OSError as exc:
        return [_error("tt-metal pin", f"failed to read {pin_path}: {exc}")]
    if pin is None:
        return [_error("tt-metal pin", f"no commit found in {pin_path}")]

    checks.append(_ok("tt-metal pin", pin))
    current = _git_output(tt_metal_root, "rev-parse", "HEAD")
    if current is None:
        checks.append(_error("tt-metal current commit", "git rev-parse failed"))
    elif current == pin:
        checks.append(_ok("tt-metal current commit", current))
    else:
        checks.append(
            _error(
                "tt-metal current commit",
                f"{current}; expected pinned tt-emule commit {pin}",
            )
        )

    has_pin = subprocess.run(
        ["git", "-C", str(tt_metal_root), "cat-file", "-e", f"{pin}^{{commit}}"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if has_pin.returncode == 0:
        checks.append(_ok("tt-metal pinned commit present", pin))
    else:
        checks.append(
            _error(
                "tt-metal pinned commit present",
                "missing from checkout; fetch full history or the pinned commit before building tt-emule",
            )
        )
    return checks


def _check_submodules(tt_metal_root: Path) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    for submodule in REQUIRED_SUBMODULES:
        completed = subprocess.run(
            ["git", "-C", str(tt_metal_root), "submodule", "status", submodule],
            check=False,
            capture_output=True,
            text=True,
        )
        output = completed.stdout.strip()
        if completed.returncode != 0:
            checks.append(_error(f"submodule:{submodule}", "git submodule status failed"))
        elif output.startswith("-"):
            checks.append(_error(f"submodule:{submodule}", "not initialized"))
        elif output.startswith("+"):
            checks.append(_error(f"submodule:{submodule}", f"checked out at unexpected commit: {output}"))
        else:
            checks.append(_ok(f"submodule:{submodule}", output))
    return checks


def _find_metalium_package(tt_metal_root: Path) -> Path | None:
    for prefix in (
        tt_metal_root / "build_emule",
        tt_metal_root / "build",
        tt_metal_root / "build_Release",
        tt_metal_root / "build_RelWithDebInfo",
        tt_metal_root / ".build" / "default",
    ):
        for config in _metalium_config_paths(prefix):
            if _is_usable_metalium_config(config):
                return prefix
    return None


def _is_usable_metalium_config(config: Path) -> bool:
    return config.exists() and (config.parent / "Metalium.cmake").exists()


def _metalium_config_paths(prefix: Path) -> tuple[Path, ...]:
    return (
        prefix / "tt-metalium-config.cmake",
        prefix / "lib" / "cmake" / "tt-metalium" / "tt-metalium-config.cmake",
        prefix / "lib64" / "cmake" / "tt-metalium" / "tt-metalium-config.cmake",
        prefix / "lib" / "cmake" / "tt-metalium" / "TT-MetaliumConfig.cmake",
        prefix / "lib64" / "cmake" / "tt-metalium" / "TT-MetaliumConfig.cmake",
        prefix / "tt_metal" / "cmake" / "TT-MetaliumConfig.cmake",
    )


def _read_pin(path: Path) -> str | None:
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped
    return None


def _git_output(root: Path, *args: str) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _ok(name: str, detail: object) -> dict[str, str]:
    return {"name": name, "status": "ok", "detail": str(detail)}


def _error(name: str, detail: object) -> dict[str, str]:
    return {"name": name, "status": "error", "detail": str(detail)}


if __name__ == "__main__":
    raise SystemExit(main())
