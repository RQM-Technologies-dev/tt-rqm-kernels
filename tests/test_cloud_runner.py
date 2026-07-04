from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_cloud_runner_check_is_safe() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/rqm_tt_cloud_runner.py", "--check"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "RQM Tenstorrent Cloud runner check" in completed.stdout
    assert "cloud API client: not implemented" in completed.stdout
    assert "paid provisioning: not implemented" in completed.stdout
    assert "observed Console status: API inference available" in completed.stdout
    assert "observed Console status: billing/usage visible" in completed.stdout
    assert "observed Console status: compute visible" in completed.stdout
    assert "resources available, no allocation observed" in completed.stdout
    assert "instances/baremetal blocked until access" in completed.stdout
    assert "capacity request path: Compute -> Resources -> Request Capacity" in completed.stdout
    assert "vscode: copy/paste run inside a granted Console VSCode/browser instance" in completed.stdout


def test_cloud_runner_vscode_mode_prints_copy_paste_instructions() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/rqm_tt_cloud_runner.py",
            "--mode",
            "vscode",
            "--print-instructions",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Tenstorrent Console VSCode/browser instance instructions" in completed.stdout
    assert "Compute -> Resources -> Request Capacity" in completed.stdout
    assert "git clone https://github.com/RQM-Technologies-dev/tt-rqm-kernels.git" in completed.stdout
    assert "TT_RQM_HARDWARE_QMUL_COMMAND" in completed.stdout
    assert "reports/tt_hardware_qmul_quickstart.json" in completed.stdout


def test_cloud_runner_vscode_mode_requires_print_instructions() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/rqm_tt_cloud_runner.py",
            "--mode",
            "vscode",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "vscode mode is copy/paste instruction-only" in completed.stderr


def test_cloud_runner_delegated_mode_prints_instructions() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/rqm_tt_cloud_runner.py",
            "--mode",
            "delegated",
            "--print-instructions",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Delegated Tenstorrent hardware validation instructions" in completed.stdout
    assert "No RQM cloud billing or provisioning is required" in completed.stdout
    assert "reports/tt_hardware_qmul_quickstart.json" in completed.stdout
    assert "reports/tt_hardware_qmul_quickstart.md" in completed.stdout
    assert "Compute -> Resources -> Request Capacity" in completed.stdout
    assert "VSCode/browser instance or SSH baremetal" in completed.stdout


def test_cloud_runner_ssh_dry_run_does_not_execute() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/rqm_tt_cloud_runner.py",
            "--mode",
            "ssh",
            "--host",
            "tenstorrent.example",
            "--remote-dir",
            "/home/tt/tt-rqm-kernels",
            "--remote-command",
            "/opt/tt/qmul_hw",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "SSH hardware run command:" in completed.stdout
    assert "ssh tenstorrent.example" in completed.stdout
    assert "Dry run only. Add --execute" in completed.stdout
    assert "Executing SSH command" not in completed.stdout


def test_cloud_runner_ssh_requires_explicit_execute_to_run() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/rqm_tt_cloud_runner.py",
            "--mode",
            "ssh",
            "--host",
            "tenstorrent.example",
            "--remote-dir",
            "/home/tt/tt-rqm-kernels",
            "--remote-command",
            "/opt/tt/qmul_hw",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Dry run only" in completed.stdout
    assert completed.stderr == ""


def test_cloud_runner_ssh_missing_required_args_fails() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/rqm_tt_cloud_runner.py", "--mode", "ssh"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "ssh mode requires --host, --remote-dir, --remote-command" in completed.stderr


def test_cloud_runner_writes_no_credentials_to_home(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["HOME"] = str(tmp_path)
    env["XDG_CONFIG_HOME"] = str(tmp_path / "config")

    subprocess.run(
        [sys.executable, "scripts/rqm_tt_cloud_runner.py", "--check"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/rqm_tt_cloud_runner.py",
            "--mode",
            "vscode",
            "--print-instructions",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/rqm_tt_cloud_runner.py",
            "--mode",
            "delegated",
            "--print-instructions",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert list(tmp_path.rglob("*")) == []
