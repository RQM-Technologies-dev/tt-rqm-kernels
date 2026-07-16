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
    assert result["h2a"] == {"status": "pre_hardware", "target_claim_level": 0}


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
            "one non-designated pilot; designated conformance is\n  absent.",
            "H2A hardware implementation: complete.",
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
