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
CLEAN_REPRODUCTION = EVIDENCE_ROOT / "h2a-clean-reproduction-20260716"
CLEAN_REPRODUCTION_INDEX = CLEAN_REPRODUCTION / "artifact-index.json"


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
        raise HamiltonianLoweringEvidenceError(
            "invalid or missing retained evidence index"
        ) from exc
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


def build_clean_reproduction_index(repo_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    root = repo_root / CLEAN_REPRODUCTION
    if not root.is_dir():
        raise HamiltonianLoweringEvidenceError("missing clean reproduction package")
    files = [
        {
            "path": path.relative_to(repo_root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != repo_root / CLEAN_REPRODUCTION_INDEX
    ]
    return {"schema": "tt-rqm-h2a-clean-reproduction-index.v1", "files": files}


def validate_clean_reproduction(repo_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    index = _load_object(repo_root / CLEAN_REPRODUCTION_INDEX)
    if index != build_clean_reproduction_index(repo_root):
        raise HamiltonianLoweringEvidenceError("clean reproduction file hash or inventory changed")
    summary = _load_object(repo_root / CLEAN_REPRODUCTION / "reproducibility.json")
    expected = {
        "repository_commit": "225cb213ae79df7acd43d6056841c3eae7b5fc40",
        "source_tree_clean": True,
        "source_bundle_sha256": "519b2b9ffb7341893aed1574604ce3c0021b9c47830ca9c297d03d69b7cf80d5",
        "tt_metal_commit": "dd2849b5bc6b7a5d38a9eafbeba31ef8d530f8d4",
        "tt_metal_tree_clean": True,
        "binary_byte_identical": True,
        "nine_case_suite_passed": True,
        "all_output_checksums_match_retained_dirty_tree_pilot": True,
        "designated": False,
        "qualification_eligible": False,
        "claim_level": None,
        "stable_benchmark": False,
        "performance_eligible": False,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise HamiltonianLoweringEvidenceError(f"clean reproduction {key} mismatch")
    builds = summary.get("builds", ())
    if len(builds) != 2 or any(
        build.get("candidate_binary_sha256")
        != "b12063fd8ff73ff7372713eeb3fbdea31c56462c94e314713909a1f07e225979"
        for build in builds
    ):
        raise HamiltonianLoweringEvidenceError("clean build identities mismatch")
    clean_pilot = repo_root / CLEAN_REPRODUCTION / "nine-case-pilot"
    clean_result = validate_pilot_package(clean_pilot)
    retained = _load_object(repo_root / PILOT / "suite-report.json")
    reproduced = _load_object(clean_pilot / "suite-report.json")
    old_checksums = [item.get("output_checksum") for item in retained["results"]]
    new_checksums = [item.get("output_checksum") for item in reproduced["results"]]
    if old_checksums != new_checksums:
        raise HamiltonianLoweringEvidenceError("clean output checksums differ from retained pilot")
    return {
        "clean_reproduction_valid": True,
        "file_count": len(index["files"]),
        "build_count": 2,
        "case_count": clean_result["case_count"],
        "outputs_byte_identical": True,
    }


def _load_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise HamiltonianLoweringEvidenceError(f"invalid retained JSON: {path.name}") from exc
    if not isinstance(payload, dict):
        raise HamiltonianLoweringEvidenceError(f"retained JSON must be an object: {path.name}")
    return payload
