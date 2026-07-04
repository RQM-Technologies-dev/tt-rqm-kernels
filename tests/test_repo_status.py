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
    assert statuses["TT-Metalium candidate"] == "source candidate present / not built"
    assert statuses["tt-emule candidate"] == "not implemented"
    assert statuses["hardware report"] == "not implemented"


def test_repo_status_text_is_maintainer_scannable() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/repo_status.py"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "CPU/PyTorch reference: implemented" in completed.stdout
    assert "TT-Metalium candidate: source candidate present / not built" in completed.stdout
    assert "hardware report: not implemented" in completed.stdout
