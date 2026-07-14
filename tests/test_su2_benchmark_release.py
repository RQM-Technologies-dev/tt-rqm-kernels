from __future__ import annotations

import copy
import re
from pathlib import Path
import subprocess
import sys

import pytest

from tt_rqm_kernels.su2_benchmark_release import (
    DEFAULT_MANIFEST,
    SU2ReleaseError,
    generate_release,
    load_manifest,
    validate_manifest,
    validate_release,
)


ROOT = Path(__file__).resolve().parents[1]


def test_su2_release_and_hardware_report_validate() -> None:
    release = validate_release(ROOT / DEFAULT_MANIFEST, repo_root=ROOT)
    assert release["claim"] == {
        "level": 1,
        "name": "qualified_first_comparison_sample",
        "public_session_count": 1,
        "stable_benchmark": False,
    }


def test_su2_artifact_tampering_is_rejected() -> None:
    release = copy.deepcopy(load_manifest(ROOT / DEFAULT_MANIFEST))
    release["artifacts"][0]["sha256"] = "0" * 64
    with pytest.raises(SU2ReleaseError, match="SHA-256 mismatch"):
        validate_manifest(release, repo_root=ROOT)


def test_su2_level_two_requires_three_sessions() -> None:
    release = copy.deepcopy(load_manifest(ROOT / DEFAULT_MANIFEST))
    release["claim"].update({"level": 2, "stable_benchmark": True})
    with pytest.raises(SU2ReleaseError, match="at least three"):
        validate_manifest(release, repo_root=ROOT)


def test_su2_measured_bandwidth_field_requires_ceiling() -> None:
    release = copy.deepcopy(load_manifest(ROOT / DEFAULT_MANIFEST))
    release["measured_bandwidth_gb_per_s"] = 42.0
    with pytest.raises(SU2ReleaseError, match="ceiling artifact"):
        validate_manifest(release, repo_root=ROOT)


def test_su2_outputs_are_byte_deterministic(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    outputs = generate_release(ROOT / DEFAULT_MANIFEST, repo_root=ROOT, destination_root=first)
    assert outputs == generate_release(ROOT / DEFAULT_MANIFEST, repo_root=ROOT, destination_root=second)
    for relative in outputs:
        assert (first / relative).read_bytes() == (second / relative).read_bytes()
        assert (ROOT / relative).read_bytes() == (first / relative).read_bytes()
        if relative.suffix == ".svg":
            svg = (first / relative).read_text()
            assert re.search(r'id="[mp][0-9a-f]{10}"', svg) is None


def test_su2_one_command_check_is_read_only() -> None:
    before = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True).stdout
    completed = subprocess.run(
        [sys.executable, "scripts/reproduce_wormhole_su2_compose.py", "--check"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    after = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True).stdout
    assert before == after
    assert "Claim Level 1" in completed.stdout
    assert "stable_benchmark=false" in completed.stdout


def test_su2_collection_requires_direct_candidate_command() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/reproduce_wormhole_su2_compose.py", "--collect", "performance"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "--collect requires --command" in completed.stderr
