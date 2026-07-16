"""Integrity validation for retained H2A development evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from tt_rqm_kernels.hamiltonian_lowering_pilot import validate_pilot_package

EVIDENCE_INDEX_SCHEMA = "tt-rqm-h2a-retained-evidence-index.v1"
EVIDENCE_ROOT = Path("benchmarks/pilots/hamiltonian-lowering-h2a")
BLOCKER = EVIDENCE_ROOT / "h2a-n300-development-blocker-20260716"
COMPARISON = EVIDENCE_ROOT / "h2a-compensated-development-20260716"
PILOT = EVIDENCE_ROOT / "h2a-compensated-n300-pilot-20260716"
INDEX = EVIDENCE_ROOT / "retained-evidence-index.json"


class HamiltonianLoweringEvidenceError(ValueError):
    """Raised when retained H2A development evidence changes."""


def build_evidence_index(repo_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    packages = []
    for relative in (BLOCKER, COMPARISON, PILOT):
        root = repo_root / relative
        if not root.is_dir():
            raise HamiltonianLoweringEvidenceError(f"missing retained package: {relative}")
        files = [
            {
                "path": path.relative_to(repo_root).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(root.rglob("*"))
            if path.is_file()
        ]
        packages.append({"path": relative.as_posix(), "files": files})
    return {"schema": EVIDENCE_INDEX_SCHEMA, "packages": packages}


def validate_retained_evidence(repo_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    try:
        index = json.loads((repo_root / INDEX).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise HamiltonianLoweringEvidenceError("invalid or missing retained evidence index") from exc
    if index != build_evidence_index(repo_root):
        raise HamiltonianLoweringEvidenceError("retained evidence file hash or inventory changed")

    blocker = _load_object(repo_root / BLOCKER / "blocker-report.json")
    if any(
        blocker.get(key) != expected
        for key, expected in {
            "designated": False,
            "qualification_eligible": False,
            "performance_eligible": False,
            "stable_benchmark": False,
            "claim_level": None,
            "pilot_started": False,
        }.items()
    ):
        raise HamiltonianLoweringEvidenceError("retained blocker claim boundary changed")
    original = blocker.get("candidate_identity", {})
    if original.get("candidate_sha256") != (
        "ca24f5253b8869ca92621e6031cc08c1d4bdafe669185e02593671a8727f3792"
    ) or original.get("source_bundle_sha256") != (
        "a307055702acd4f370d80ee8fa9a59a48e81f209d3174e9d5358d61e544bdeed"
    ):
        raise HamiltonianLoweringEvidenceError("original development identity changed")

    comparison = _load_object(repo_root / COMPARISON / "comparison-report.json")
    if any(
        comparison.get(key) != expected
        for key, expected in {
            "designated": False,
            "qualification_eligible": False,
            "performance_eligible": False,
            "stable_benchmark": False,
            "claim_level": None,
        }.items()
    ):
        raise HamiltonianLoweringEvidenceError("comparison claim boundary changed")
    compensated = comparison.get("compensated_candidate", {})
    if compensated.get("candidate_sha256") != (
        "433e74b827d2cf9a7a790a6c9d7bb3917fc1fed3915ec384de0486cdc014d306"
    ) or compensated.get("source_bundle_sha256") != (
        "7fb65217e05139bf035952ebeb34602d49e5f1772b8dec4c336b7a296e1fba2f"
    ):
        raise HamiltonianLoweringEvidenceError("compensated development identity changed")
    pilot = validate_pilot_package(repo_root / PILOT)
    return {
        "retained_evidence_valid": True,
        "package_count": len(index["packages"]),
        "file_count": sum(len(package["files"]) for package in index["packages"]),
        "pilot_passed": pilot["pilot_passed"],
    }


def _load_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise HamiltonianLoweringEvidenceError(f"invalid retained JSON: {path.name}") from exc
    if not isinstance(payload, dict):
        raise HamiltonianLoweringEvidenceError(f"retained JSON must be an object: {path.name}")
    return payload
