from __future__ import annotations

from pathlib import Path
import shutil

import pytest

from tt_rqm_kernels.hamiltonian_lowering_evidence import (
    COMPARISON,
    EVIDENCE_ROOT,
    HamiltonianLoweringEvidenceError,
    INDEX,
    validate_retained_evidence,
)

ROOT = Path(__file__).resolve().parents[1]


def test_retained_h2a_evidence_hashes_and_claim_boundaries_validate() -> None:
    assert validate_retained_evidence(ROOT) == {
        "retained_evidence_valid": True,
        "package_count": 3,
        "file_count": 118,
        "pilot_passed": True,
    }


def test_retained_h2a_evidence_rejects_tampering(tmp_path: Path) -> None:
    shutil.copytree(ROOT / EVIDENCE_ROOT, tmp_path / EVIDENCE_ROOT)
    target = tmp_path / COMPARISON / "comparison-report.json"
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(HamiltonianLoweringEvidenceError, match="hash or inventory"):
        validate_retained_evidence(tmp_path)

    assert (tmp_path / INDEX).is_file()
