from __future__ import annotations

from pathlib import Path
import shutil

import pytest

from tt_rqm_kernels.repository_claims import (
    RepositoryClaimsError,
    STATUS_FILES,
    _load_status_documents,
    _validate_status_surfaces,
    validate_repository_claims,
)

ROOT = Path(__file__).resolve().parents[1]


def test_repository_claims_agree_with_protected_releases() -> None:
    result = validate_repository_claims(ROOT)
    assert result["qmul"]["stable_benchmark"] is True
    assert result["su2"]["scope"] == "fused_only"
    assert result["h2a"] == {
        "status": "claim_level_0",
        "level": 0,
        "name": "silicon_conformance",
        "public_session_count": 1,
        "stable_benchmark": False,
        "performance_eligible": False,
    }
    assert result["h2b"] == {
        "status": "source_foundation",
        "stable_benchmark": False,
        "performance_eligible": False,
        "claim_level": None,
        "hardware_run": False,
    }


@pytest.mark.parametrize(
    ("relative", "old", "new", "message"),
    [
        (
            "plan.md",
            "SU2ComposeBench H1: Claim Level 2",
            "SU2ComposeBench H1: Claim Level 1",
            "plan status marker missing",
        ),
        (
            "plan.md",
            "historical H1 v2 fused/unfused campaign is retained and non-qualifying",
            "v3 fused/unfused stable release",
            "plan status marker missing",
        ),
        (
            "plan.md",
            "Every individual qmul and H1 source-session report remains\n  `stable_benchmark=false`",
            "Every individual qmul and H1 source-session report is\n  `stable_benchmark=true`",
            "plan status marker missing",
        ),
        (
            "plan.md",
            "H2A device-side coefficient lowering: Claim Level 0 silicon conformance from\n  one designated N300 device-0 session; `stable_benchmark=false` and\n  `performance_eligible=false`.",
            "H2A hardware implementation: complete.",
            "plan status marker missing",
        ),
        (
            "plan.md",
            "H2B integration foundation: CPU/reference API and fail-closed protocol are\n  implemented; a two-program TT-Metal candidate feeds protected fused H1 from\n  a device-DRAM intermediate. Hardware has not yet been run;\n  `stable_benchmark=false`, `performance_eligible=false`, `claim_level=null`.",
            "H2B claim level 0 established.",
            "plan status marker missing",
        ),
    ],
)
def test_repository_claims_reject_status_regressions(
    tmp_path: Path, relative: str, old: str, new: str, message: str
) -> None:
    _copy_status_files(tmp_path)
    path = tmp_path / relative
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    with pytest.raises(RepositoryClaimsError, match=message):
        _validate_status_surfaces(_load_status_documents(tmp_path))


def _copy_status_files(destination: Path) -> None:
    for relative in STATUS_FILES:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
