from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from tt_rqm_kernels.hamiltonian_evolution_pilot import (
    HamiltonianEvolutionPilotError,
    validate_pilot_package,
)

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "benchmarks/pilots/hamiltonian-evolution-h2b/h2b-n300-pilot-20260716-session-1"


def test_retained_first_pilot_is_valid_and_failed() -> None:
    assert validate_pilot_package(PILOT, ROOT) == {
        "package_valid": True,
        "pilot_passed": False,
        "case_count": 20,
    }
    suite = json.loads((PILOT / "suite-report.json").read_text())
    assert len(suite["results"]) == 20
    assert all(item["attempt_count"] == 1 for item in suite["results"])
    assert all(item["retry_count"] == 0 for item in suite["results"])
    assert all(item["candidate_completed"] is False for item in suite["results"])


def test_offline_qualifier_rejects_retry_or_reordered_case(tmp_path: Path) -> None:
    copied = tmp_path / "pilot"
    shutil.copytree(PILOT, copied)
    manifest_path = copied / "pilot-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["retries"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(HamiltonianEvolutionPilotError, match="retries"):
        validate_pilot_package(copied, ROOT)

    shutil.rmtree(copied)
    shutil.copytree(PILOT, copied)
    suite_path = copied / "suite-report.json"
    suite = json.loads(suite_path.read_text())
    suite["results"] = list(reversed(suite["results"]))
    suite_path.write_text(json.dumps(suite), encoding="utf-8")
    with pytest.raises(HamiltonianEvolutionPilotError, match="missing or reordered"):
        validate_pilot_package(copied, ROOT)
