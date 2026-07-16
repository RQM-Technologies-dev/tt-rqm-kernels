from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from tt_rqm_kernels.hamiltonian_lowering_release import (
    HamiltonianLoweringReleaseError,
    QUALIFICATION_PATH,
    RELEASE_MANIFEST_PATH,
    SESSION_ROOT,
    SUMMARY_PATH,
    build_processed_summary,
    validate_manifest,
    validate_release,
)

ROOT = Path(__file__).resolve().parents[1]


def test_public_h2a_claim_level_0_release_validates_and_regenerates() -> None:
    release = validate_release(ROOT / RELEASE_MANIFEST_PATH, repo_root=ROOT)
    assert release["claim"] == {
        "level": 0,
        "name": "silicon_conformance",
        "public_session_count": 1,
        "stable_benchmark": False,
        "performance_eligible": False,
    }
    assert release["sessions"][0]["qualification_passed"] is True
    assert release["origin_package"]["file_count"] == 46


def test_processed_summary_is_exactly_reproducible() -> None:
    release = _load(RELEASE_MANIFEST_PATH)
    session = _load(SESSION_ROOT / "session-manifest.json")
    expected = build_processed_summary(release, session, ROOT)
    assert _load(SUMMARY_PATH) == expected
    assert expected["claim"]["stable_benchmark"] is False
    assert expected["qualification"]["qualification_passed"] is True


def test_source_session_and_all_case_reports_remain_unpromoted() -> None:
    session = _load(SESSION_ROOT / "session-manifest.json")
    assert session["claim_level"] is None
    assert session["stable_benchmark"] is False
    assert session["performance_eligible"] is False
    for result in session["results"]:
        report = _load(SESSION_ROOT / result["report"])
        assert report["claim_level"] is None
        assert report["stable_benchmark"] is False
        assert report["performance_eligible"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("level", 1),
        ("stable_benchmark", True),
        ("performance_eligible", True),
        ("public_session_count", 2),
    ],
)
def test_release_rejects_claim_promotion(field: str, value: object) -> None:
    release = deepcopy(_load(RELEASE_MANIFEST_PATH))
    release["claim"][field] = value
    with pytest.raises(HamiltonianLoweringReleaseError, match="Claim Level 0 boundary"):
        validate_manifest(release, repo_root=ROOT)


def test_release_rejects_missing_hash_bound_artifact() -> None:
    release = deepcopy(_load(RELEASE_MANIFEST_PATH))
    release["artifacts"].pop()
    with pytest.raises(HamiltonianLoweringReleaseError, match="artifact set changed"):
        validate_manifest(release, repo_root=ROOT)


def test_qualification_is_conformance_only() -> None:
    qualification = _load(QUALIFICATION_PATH)
    assert qualification == {
        "schema": "tt-rqm-hamiltonian-lowering-qualification.v1",
        "qualification_passed": True,
        "target_claim_level": 0,
        "claim_level": None,
        "stable_benchmark": False,
        "performance_eligible": False,
        "release_created": False,
    }


def _load(relative: Path) -> dict[str, object]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))
