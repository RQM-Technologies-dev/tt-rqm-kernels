from __future__ import annotations

from pathlib import Path
import shutil

import pytest

from tt_rqm_kernels.hamiltonian_lowering_evidence import (
    CLEAN_REPRODUCTION,
    COMPARISON,
    EVIDENCE_ROOT,
    HamiltonianLoweringEvidenceError,
    INDEX,
    validate_retained_evidence,
    validate_clean_reproduction,
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


def test_clean_reproduction_builds_and_outputs_validate() -> None:
    assert validate_clean_reproduction(ROOT) == {
        "clean_reproduction_valid": True,
        "file_count": 61,
        "build_count": 2,
        "case_count": 9,
        "outputs_byte_identical": True,
    }


def test_clean_reproduction_rejects_tampering(tmp_path: Path) -> None:
    shutil.copytree(ROOT / EVIDENCE_ROOT, tmp_path / EVIDENCE_ROOT)
    target = tmp_path / CLEAN_REPRODUCTION / "reproducibility.json"
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(HamiltonianLoweringEvidenceError, match="hash or inventory"):
        validate_clean_reproduction(tmp_path)
