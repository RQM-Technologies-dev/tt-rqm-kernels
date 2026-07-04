from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_tt_emule_environment_check_unavailable_is_clear() -> None:
    env = _without_tt_emule_env()
    completed = subprocess.run(
        [
            sys.executable,
            "experimental/tt_emule_qmul/check_environment.py",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 2
    assert "tt-emule environment unavailable" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_tt_emule_environment_check_accepts_fake_roots(
    tmp_path: Path,
) -> None:
    tt_metal = tmp_path / "tt-metal"
    tt_emule = tmp_path / "tt-emule"
    (tt_metal / "tt_metal").mkdir(parents=True)
    (tt_metal / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.24)\n")
    (tt_emule / "include" / "tt_emule").mkdir(parents=True)
    (tt_emule / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.24)\n")

    completed = subprocess.run(
        [
            sys.executable,
            "experimental/tt_emule_qmul/check_environment.py",
            "--skip-platform-check",
            "--tt-metal-root",
            str(tt_metal),
            "--tt-emule-root",
            str(tt_emule),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "tt-metal root detected" in completed.stdout
    assert "tt-emule root detected" in completed.stdout
    assert "does not run a kernel" in completed.stdout


def _without_tt_emule_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("TT_METAL_HOME", None)
    env.pop("TT_METALIUM_HOME", None)
    env.pop("TT_EMULE_HOME", None)
    return env
