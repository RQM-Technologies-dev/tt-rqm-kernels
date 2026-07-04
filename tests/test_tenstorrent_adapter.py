from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tt_rqm_kernels.backends.tenstorrent.availability import (
    DEFAULT_HARDWARE_COMMAND_ENV,
    check_readiness,
    inspect_hardware_command,
    resolve_execution_path,
)
from tt_rqm_kernels.backends.tenstorrent.qmul_external import (
    TenstorrentAdapterError,
    run_external_qmul_inputs,
)
from tt_rqm_kernels.backends.tenstorrent.report import (
    ReportLabelError,
    validate_external_qmul_label,
)


FAST_EXTERNAL_QMUL = f"{sys.executable} tests/fixtures/qmul_external_fast.py"


def test_tenstorrent_readiness_missing_environment_is_explicit(tmp_path: Path) -> None:
    readiness = check_readiness(
        repo_root=tmp_path,
        report_dir=tmp_path / "reports",
        env={},
    )

    assert readiness.tt_metal_home is None
    assert readiness.tt_emule_home is None
    assert readiness.emule_candidate_script_present is False
    assert readiness.emule_ready is False
    assert readiness.hardware_ready is False
    details = {item.name: item.detail for item in readiness.checks}
    assert "TT_METAL_HOME unset" in details["TT_METAL_HOME"]
    assert "TT_EMULE_HOME unset" in details["TT_EMULE_HOME"]


def test_resolve_hardware_path_requires_configured_command() -> None:
    path = resolve_execution_path("hardware", env={})

    assert path.available is False
    assert path.command is None
    assert path.execution_label == "hardware"
    assert DEFAULT_HARDWARE_COMMAND_ENV in path.reason


def test_hardware_command_preflight_rejects_missing_executable() -> None:
    preflight = inspect_hardware_command("/definitely/missing/rqm_qmul_hw")

    assert preflight.available is False
    assert "executable not found" in preflight.reason


def test_hardware_command_preflight_accepts_safe_fixture_command() -> None:
    preflight = inspect_hardware_command(f"{sys.executable} --version")

    assert preflight.available is True
    assert preflight.executable is not None
    assert "hardware command executable found" in preflight.reason


def test_hardware_command_preflight_rejects_emule_command() -> None:
    preflight = inspect_hardware_command(
        "bash experimental/tt_metalium_qmul/run_candidate_docker.sh"
    )

    assert preflight.available is False
    assert "tt-emule/emulation" in preflight.reason


def test_report_label_validation_keeps_emulation_out_of_hardware() -> None:
    assert validate_external_qmul_label("emulation", command="bash run_candidate_docker.sh") == "emulation"
    assert validate_external_qmul_label("hardware", command="/opt/tt/qmul_hw") == "hardware"

    with pytest.raises(ReportLabelError, match="tt-emule"):
        validate_external_qmul_label("hardware", command="bash experimental/tt_metalium_qmul/run_candidate_docker.sh")
    with pytest.raises(ReportLabelError, match="tt-lang-sim"):
        validate_external_qmul_label("simulator")


def test_qmul_external_adapter_requires_command() -> None:
    a = torch.zeros((2, 4), dtype=torch.float32)
    b = torch.zeros((2, 4), dtype=torch.float32)

    with pytest.raises(TenstorrentAdapterError, match="command is not configured"):
        run_external_qmul_inputs(a, b, command=None)


def test_qmul_external_adapter_runs_fixture_command() -> None:
    a = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.5, 0.25, -0.5, 0.75]],
        dtype=torch.float32,
    )
    b = torch.tensor(
        [[0.25, 0.5, -0.75, 1.0], [1.0, -0.5, 0.5, -0.25]],
        dtype=torch.float32,
    )

    run = run_external_qmul_inputs(a, b, command=FAST_EXTERNAL_QMUL, iterations=1)

    assert run.device == "cpu/python-test-fixture"
    assert run.max_abs_error < 1e-6
    assert run.rms_error < 1e-6
    assert run.throughput > 0.0


def test_quickstart_check_path_is_ci_safe() -> None:
    env = os.environ.copy()
    env.pop(DEFAULT_HARDWARE_COMMAND_ENV, None)
    completed = subprocess.run(
        [sys.executable, "scripts/rqm_tt_quickstart.py", "--check"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert "RQM Tenstorrent qmul quickstart readiness" in completed.stdout
    assert "hardware mode ready:" in completed.stdout
    assert "Traceback" not in completed.stderr


def test_quickstart_hardware_mode_without_command_fails_cleanly() -> None:
    env = os.environ.copy()
    env.pop(DEFAULT_HARDWARE_COMMAND_ENV, None)
    completed = subprocess.run(
        [sys.executable, "scripts/rqm_tt_quickstart.py", "--mode", "hardware"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 2
    assert "hardware command is not configured" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_quickstart_emule_stable_benchmark_fails_cleanly() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/rqm_tt_quickstart.py",
            "--mode",
            "emule",
            "--stable-benchmark",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "stable benchmark reports are not allowed for --mode emule" in completed.stderr
    assert "Traceback" not in completed.stderr
