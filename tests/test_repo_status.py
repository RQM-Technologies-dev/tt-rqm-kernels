from __future__ import annotations

import json
import subprocess
import sys


def test_repo_status_json_reports_current_gaps() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/repo_status.py", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["schema"] == "tt-rqm-repo-status.v1"
    statuses = {item["name"]: item["status"] for item in payload["items"]}
    assert statuses["CPU/PyTorch reference"] == "implemented"
    assert statuses["StructuredBench smoke"] == "implemented"
    assert statuses["external-qmul harness"] == "implemented"
    assert statuses["TT-Metalium candidate"] == "experimental source candidate present"
    assert statuses["tt-emule candidate"] == "emulation report present"
    assert statuses["hardware report"] == "hardware conformance report present"
    assert statuses["Stage B hardware report"] == "first hardware sample present"


def test_repo_status_text_is_maintainer_scannable() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/repo_status.py"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "CPU/PyTorch reference: implemented" in completed.stdout
    assert "TT-Metalium candidate: experimental source candidate present" in completed.stdout
    assert "tt-emule candidate: emulation report present" in completed.stdout
    assert "hardware report: hardware conformance report present" in completed.stdout
    assert "Stage B hardware report: first hardware sample present" in completed.stdout
    assert "not performance-eligible" in completed.stdout
    assert "not an acceleration claim" in completed.stdout
