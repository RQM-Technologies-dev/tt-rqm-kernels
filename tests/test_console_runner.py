from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_console_runner_check_exits_zero() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/rqm_tt_console_runner.py", "--check"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "RQM Tenstorrent Console runner check" in completed.stdout
    assert "quickstart script: present" in completed.stdout
    assert "billing/provisioning: not implemented" in completed.stdout
    assert "network calls: not performed" in completed.stdout
    assert "credentials required: no" in completed.stdout
    assert "API inference available: yes" in completed.stdout
    assert "billing/usage visible: yes" in completed.stdout
    assert "compute visible: yes" in completed.stdout
    assert "resources page available: yes" in completed.stdout
    assert "dedicated hardware allocation: none observed" in completed.stdout
    assert "instances access: blocked until access is granted" in completed.stdout
    assert "baremetal access: blocked until access is granted" in completed.stdout
    assert "capacity request path: Compute -> Resources -> Request Capacity" in completed.stdout


def test_console_runner_browser_steps_include_billing_stop() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/rqm_tt_console_runner.py",
            "--print-browser-steps",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Tenstorrent Cloud Console browser workflow" in completed.stdout
    assert "Confirm Usage and Billing are visible" in completed.stdout
    assert "Compute -> Resources" in completed.stdout
    assert "Request Capacity" in completed.stdout
    assert "Instances: managed VSCode/browser shell" in completed.stdout
    assert "Baremetal: SSH run" in completed.stdout
    assert "No cloud resources are created" in completed.stdout
    assert "No credentials are requested or stored" in completed.stdout
    assert "No capacity request is submitted" in completed.stdout


def test_console_runner_copy_paste_includes_hardware_command_env() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/rqm_tt_console_runner.py",
            "--print-copy-paste-commands",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "TT_RQM_HARDWARE_QMUL_COMMAND" in completed.stdout
    assert "python scripts/rqm_tt_quickstart.py" in completed.stdout
    assert "reports/tt_hardware_qmul_quickstart.json" in completed.stdout
    assert "Compute -> Resources -> Request Capacity" in completed.stdout


def test_console_runner_open_url_text_only() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/rqm_tt_console_runner.py", "--open-url-text"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == "https://console.tenstorrent.com/"
    assert completed.stderr == ""


def test_console_runner_writes_no_credentials_to_home(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["HOME"] = str(tmp_path)
    env["XDG_CONFIG_HOME"] = str(tmp_path / "config")

    subprocess.run(
        [sys.executable, "scripts/rqm_tt_console_runner.py", "--check"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/rqm_tt_console_runner.py",
            "--print-copy-paste-commands",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert list(tmp_path.rglob("*")) == []
