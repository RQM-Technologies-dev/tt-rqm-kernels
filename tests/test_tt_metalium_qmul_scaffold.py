from __future__ import annotations

import json
import os
import subprocess
import sys


def test_tt_metalium_environment_check_missing_sdk_is_clear() -> None:
    env = _without_tt_metal_env()
    completed = subprocess.run(
        [
            sys.executable,
            "experimental/tt_metalium_qmul/check_environment.py",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 2
    assert "TT-Metalium SDK unavailable" in completed.stderr
    assert "TT_METAL_HOME" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_tt_metalium_placeholder_without_external_env_fails_cleanly() -> None:
    env = _without_external_qmul_env()
    completed = subprocess.run(
        [
            sys.executable,
            "experimental/tt_metalium_qmul/run_candidate.py",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 2
    assert "external-qmul environment missing" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_tt_metalium_build_placeholder_missing_sdk_is_clear() -> None:
    env = _without_tt_metal_env()
    completed = subprocess.run(
        [
            sys.executable,
            "experimental/tt_metalium_qmul/build_candidate.py",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 2
    assert "TT-Metalium SDK unavailable" in completed.stderr
    assert "Build stopped before configuring the candidate" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_tt_metalium_validation_wrapper_accepts_reference_command() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "experimental/tt_metalium_qmul/validate_candidate.py",
            "--candidate-command",
            "python scripts/qmul_external_reference.py",
            "--items",
            "32",
            "--iters",
            "1",
            "--warmup",
            "0",
            "--format",
            "json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(completed.stdout)
    assert report["schema"] == "structuredbench.v1"
    assert report["backend"] == "external-qmul"
    assert report["protocol"] == "tt-rqm-external-qmul.v1"
    assert report["results"][0]["structured_shape"] == "[32, 4]"
    assert report["results"][0]["scalar_reference_max_abs_error"] < 1e-4


def test_tt_metalium_validation_wrapper_placeholder_fails_without_sdk() -> None:
    env = _without_tt_metal_env()
    completed = subprocess.run(
        [
            sys.executable,
            "experimental/tt_metalium_qmul/validate_candidate.py",
            "--items",
            "32",
            "--iters",
            "1",
            "--warmup",
            "0",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode != 0
    assert "TT-Metalium SDK unavailable" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert "No out.bin or metrics.json was written" not in completed.stdout
    assert "hardware performance" not in completed.stdout.lower()


def test_tt_metalium_source_candidate_files_exist() -> None:
    assert os.path.exists("experimental/tt_metalium_qmul/CMakeLists.txt")
    assert os.path.exists("experimental/tt_metalium_qmul/src/qmul_candidate.cpp")
    assert os.path.exists("experimental/tt_metalium_qmul/kernels/qmul_riscv.cpp")


def _without_tt_metal_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("TT_METAL_HOME", None)
    env.pop("TT_METALIUM_HOME", None)
    return env


def _without_external_qmul_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("TT_RQM_EXTERNAL_QMUL_DIR", None)
    env.pop("TT_RQM_EXTERNAL_QMUL_MANIFEST", None)
    return env
