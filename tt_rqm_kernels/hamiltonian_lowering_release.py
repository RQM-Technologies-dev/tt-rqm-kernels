"""Hash-bound public Claim Level 0 release for H2A conformance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

from tt_rqm_kernels.hamiltonian_lowering_benchmark import CASE_IDS
from tt_rqm_kernels.hamiltonian_lowering_designated import (
    CANDIDATE_SHA256,
    COMPILER_IDENTITY,
    IMPLEMENTATION_COMMIT,
    MANIFEST_PATH as DESIGNATED_MANIFEST_PATH,
    RUNTIME_IDENTITY,
    SOURCE_BUNDLE_SHA256,
    TT_METAL_COMMIT,
    qualify_session,
    validate_designated_manifest,
)

RELEASE_SCHEMA = "tt-rqm-hamiltonian-lowering-release.v1"
SUMMARY_SCHEMA = "tt-rqm-hamiltonian-lowering-release-summary.v1"
RELEASE_MANIFEST_PATH = Path("benchmarks/manifests/wormhole-hamiltonian-lowering.json")
SESSION_ID = "h2a-designated-conformance-n300-20260716-session-1"
SESSION_ROOT = Path("benchmarks/raw/hamiltonian-lowering-h2a") / SESSION_ID
GOVERNANCE_ROOT = Path("benchmarks/raw/hamiltonian-lowering-h2a") / f"{SESSION_ID}-governance"
QUALIFICATION_PATH = Path(
    "benchmarks/processed/wormhole-hamiltonian-lowering-h2a-qualification.json"
)
SUMMARY_PATH = Path("benchmarks/processed/wormhole-hamiltonian-lowering-h2a-summary.json")
GOVERNANCE_COMMIT = "a8ee3dfa02eea3e924f625bc7e7df18cb94ce5e4"
ORIGIN_PACKAGE_TAR_SHA256 = "c68151223dbf3d789635338d641a4211334719b4a8b17c7ca7701d6b319fe746"
NONCLAIMS = {
    "no_performance_claim",
    "no_stability_claim",
    "no_acceleration_claim",
    "no_cpu_comparison",
    "no_measured_bandwidth_claim",
    "no_energy_claim",
    "no_dual_device_claim",
    "no_H2B_claim",
    "no_Tenstorrent_endorsement",
    "no_inheritance_from_H1",
}


class HamiltonianLoweringReleaseError(ValueError):
    """Raised when public H2A evidence is incomplete or inconsistent."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_release_manifest(repo_root: Path) -> dict[str, Any]:
    """Build the release manifest from fixed, already-retained evidence."""

    root = repo_root.resolve()
    artifact_specs = (
        ("benchmarks/manifests/hamiltonian-lowering-h2a-preregistration.json", "preregistration"),
        (DESIGNATED_MANIFEST_PATH.as_posix(), "frozen-designated-contract"),
        (
            "benchmarks/manifests/hamiltonian-lowering-h2a-source-manifest.json",
            "candidate-source-manifest",
        ),
        (
            "benchmarks/inputs/hamiltonian-lowering-h2a-designated-conformance/input-manifest.json",
            "frozen-input-manifest",
        ),
        ((SESSION_ROOT / "session-manifest.json").as_posix(), "designated-session-manifest"),
        (
            (GOVERNANCE_ROOT / "session-file-inventory.sha256").as_posix(),
            "complete-session-file-inventory",
        ),
        ((GOVERNANCE_ROOT / "preflight.json").as_posix(), "preflight-and-device-health"),
        (
            (GOVERNANCE_ROOT / "cpu-reference-validation.txt").as_posix(),
            "cpu-reference-validation",
        ),
        ((GOVERNANCE_ROOT / "collection-invoked-at.txt").as_posix(), "collection-start"),
        ((GOVERNANCE_ROOT / "collection-returned-at.txt").as_posix(), "collection-end"),
        ((GOVERNANCE_ROOT / "collector.exit-code.txt").as_posix(), "collector-exit-code"),
        ((GOVERNANCE_ROOT / "collector.stdout.txt").as_posix(), "collector-stdout"),
        ((GOVERNANCE_ROOT / "collector.stderr.txt").as_posix(), "collector-stderr"),
        (QUALIFICATION_PATH.as_posix(), "offline-qualification"),
    )
    artifacts = [
        {"path": relative, "role": role, "sha256": sha256_file(root / relative)}
        for relative, role in artifact_specs
    ]
    return {
        "schema": RELEASE_SCHEMA,
        "benchmark_id": "wormhole-hamiltonian-lowering-h2a-fp32",
        "title": "H2A Device-Side Hamiltonian Coefficient Lowering on Tenstorrent Wormhole",
        "primary_report": (SESSION_ROOT / "session-manifest.json").as_posix(),
        "artifacts": artifacts,
        "provenance": {
            "governance_commit": GOVERNANCE_COMMIT,
            "candidate_source_commit": IMPLEMENTATION_COMMIT,
            "source_bundle_sha256": SOURCE_BUNDLE_SHA256,
            "candidate_binary_sha256": CANDIDATE_SHA256,
            "tt_metal_commit": TT_METAL_COMMIT,
            "compiler_identity": COMPILER_IDENTITY,
            "runtime_identity": RUNTIME_IDENTITY,
        },
        "claim": {
            "level": 0,
            "name": "silicon_conformance",
            "public_session_count": 1,
            "stable_benchmark": False,
            "performance_eligible": False,
        },
        "sessions": [
            {
                "id": SESSION_ID,
                "session_root": SESSION_ROOT.as_posix(),
                "session_manifest": (SESSION_ROOT / "session-manifest.json").as_posix(),
                "artifact_inventory": (
                    GOVERNANCE_ROOT / "session-file-inventory.sha256"
                ).as_posix(),
                "qualification": QUALIFICATION_PATH.as_posix(),
                "qualification_passed": True,
                "hardware_execution": True,
                "attempts_per_case": 1,
                "retries": 0,
                "replacement_results": 0,
            }
        ],
        "qualification_gates": {
            "frozen_case_order": list(CASE_IDS),
            "whole_output_atol": 0.0001,
            "whole_output_rtol": 0.0001,
            "zero_failing_values": True,
            "zero_nonfinite_values": True,
            "one_wormhole_device": True,
            "device_id": 0,
            "core_count": 1,
            "one_attempt_per_case": True,
            "no_retry_or_replacement": True,
        },
        "origin_package": {
            "file_count": 46,
            "tar_sha256": ORIGIN_PACKAGE_TAR_SHA256,
            "transferred_inventory_byte_identical": True,
        },
        "nonclaims": sorted(NONCLAIMS),
        "processed_output": SUMMARY_PATH.as_posix(),
    }


def validate_release(
    manifest_path: Path = RELEASE_MANIFEST_PATH,
    *,
    repo_root: Path | None = None,
    verify_generated: bool = True,
) -> dict[str, Any]:
    root = (repo_root or Path.cwd()).resolve()
    manifest = _load_json(_resolve(root, manifest_path))
    validate_manifest(manifest, repo_root=root)
    if verify_generated:
        with tempfile.TemporaryDirectory(prefix="tt-rqm-h2a-release-") as temp:
            outputs = generate_release(
                manifest_path=manifest_path,
                repo_root=root,
                destination_root=Path(temp),
            )
            for relative in outputs:
                _require((root / relative).is_file(), f"missing generated H2A file: {relative}")
                _require(
                    (root / relative).read_bytes() == (Path(temp) / relative).read_bytes(),
                    f"stale generated H2A file: {relative}",
                )
    return manifest


def validate_manifest(manifest: Mapping[str, Any], *, repo_root: Path) -> None:
    root = repo_root.resolve()
    _require(manifest.get("schema") == RELEASE_SCHEMA, "H2A release schema mismatch")
    _require(
        manifest.get("benchmark_id") == "wormhole-hamiltonian-lowering-h2a-fp32",
        "H2A benchmark ID mismatch",
    )
    artifacts = manifest.get("artifacts")
    _require(isinstance(artifacts, list) and artifacts, "H2A release requires artifacts")
    by_path: dict[str, Mapping[str, Any]] = {}
    roles: set[str] = set()
    for artifact in artifacts:
        _require(isinstance(artifact, dict), "H2A artifact must be an object")
        relative = _text(artifact, "path")
        role = _text(artifact, "role")
        expected = _text(artifact, "sha256")
        _require(
            re.fullmatch(r"[0-9a-f]{64}", expected) is not None, f"invalid SHA-256: {relative}"
        )
        _require(relative not in by_path, f"duplicate H2A artifact: {relative}")
        _require(role not in roles, f"duplicate H2A artifact role: {role}")
        _require((root / relative).is_file(), f"missing H2A artifact: {relative}")
        _require(
            sha256_file(root / relative) == expected, f"H2A artifact hash mismatch: {relative}"
        )
        by_path[relative] = artifact
        roles.add(role)
    expected_manifest = build_release_manifest(root)
    _require(
        manifest.get("artifacts") == expected_manifest["artifacts"], "H2A artifact set changed"
    )

    validate_designated_manifest(root / DESIGNATED_MANIFEST_PATH, root)
    _validate_session_inventory(root)
    session = _validate_session(root)
    _validate_governance_evidence(root, session)
    qualification = _load_json(root / QUALIFICATION_PATH)
    recomputed = qualify_session(
        root / SESSION_ROOT,
        root / DESIGNATED_MANIFEST_PATH,
        root,
    )
    _require(qualification == recomputed, "H2A qualification is not reproducible")
    _require(qualification.get("qualification_passed") is True, "H2A qualification did not pass")
    _require(qualification.get("release_created") is False, "qualifier must not create releases")

    claim = manifest.get("claim")
    _require(
        claim
        == {
            "level": 0,
            "name": "silicon_conformance",
            "public_session_count": 1,
            "stable_benchmark": False,
            "performance_eligible": False,
        },
        "H2A Claim Level 0 boundary changed",
    )
    _require(
        manifest.get("provenance") == expected_manifest["provenance"],
        "H2A release provenance mismatch",
    )
    _require(
        manifest.get("sessions") == expected_manifest["sessions"],
        "H2A public session contract mismatch",
    )
    _require(
        manifest.get("qualification_gates") == expected_manifest["qualification_gates"],
        "H2A qualification gates changed",
    )
    _require(
        manifest.get("origin_package") == expected_manifest["origin_package"],
        "H2A origin package mismatch",
    )
    _require(set(manifest.get("nonclaims", ())) == NONCLAIMS, "H2A release nonclaims changed")
    _require(
        manifest.get("processed_output") == SUMMARY_PATH.as_posix(), "H2A processed output mismatch"
    )


def generate_release(
    manifest_path: Path = RELEASE_MANIFEST_PATH,
    *,
    repo_root: Path | None = None,
    destination_root: Path | None = None,
) -> tuple[Path, ...]:
    root = (repo_root or Path.cwd()).resolve()
    manifest = _load_json(_resolve(root, manifest_path))
    session = _load_json(root / SESSION_ROOT / "session-manifest.json")
    output_root = (destination_root or root).resolve()
    target = output_root / SUMMARY_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(build_processed_summary(manifest, session, root), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return (SUMMARY_PATH,)


def build_processed_summary(
    manifest: Mapping[str, Any], session: Mapping[str, Any], repo_root: Path
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for result in session["results"]:
        case_id = result["case_id"]
        report = _load_json(repo_root / SESSION_ROOT / "cases" / case_id / "report.json")
        metadata = report["candidate_metrics"]["candidate_metadata"]
        cases.append(
            {
                "case_id": case_id,
                "B": report["B"],
                "K": report["K"],
                "input_hashes": report["input_hashes"],
                "correctness": report["correctness"],
                "device_create_count": metadata["device_create_count"],
                "device_close_count": metadata["device_close_count"],
                "output_checksum": result["checksum"],
                "stable_benchmark": False,
                "performance_eligible": False,
            }
        )
    maxima = {
        field: max(float(case["correctness"][field]) for case in cases)
        for field in (
            "max_rotor_absolute_error",
            "max_phase_absolute_error",
            "rotor_norm_drift",
            "phase_norm_drift",
            "complex_matrix_reconstruction_error",
        )
    }
    return {
        "schema": SUMMARY_SCHEMA,
        "benchmark_id": manifest["benchmark_id"],
        "claim": manifest["claim"],
        "session_id": session["session_id"],
        "hardware_execution": True,
        "case_order": list(CASE_IDS),
        "provenance": manifest["provenance"],
        "qualification": {
            "qualification_passed": True,
            "path": QUALIFICATION_PATH.as_posix(),
        },
        "maxima": maxima,
        "cases": cases,
        "nonclaims": manifest["nonclaims"],
    }


def _validate_session_inventory(root: Path) -> None:
    inventory_path = root / GOVERNANCE_ROOT / "session-file-inventory.sha256"
    lines = inventory_path.read_text(encoding="utf-8").splitlines()
    _require(len(lines) == 46, "H2A session inventory must contain 46 files")
    entries: dict[str, str] = {}
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  \./(.+)", line)
        _require(match is not None, "malformed H2A session inventory line")
        expected, relative = match.groups()
        _require(relative not in entries, f"duplicate H2A session inventory path: {relative}")
        _require(
            ".." not in Path(relative).parts and not Path(relative).is_absolute(),
            "unsafe H2A inventory path",
        )
        entries[relative] = expected
    actual = {
        path.relative_to(root / SESSION_ROOT).as_posix(): sha256_file(path)
        for path in sorted((root / SESSION_ROOT).rglob("*"))
        if path.is_file()
    }
    _require(actual == entries, "H2A session inventory or file hash changed")


def _validate_session(root: Path) -> dict[str, Any]:
    session = _load_json(root / SESSION_ROOT / "session-manifest.json")
    expected = {
        "schema": "tt-rqm-hamiltonian-lowering-designated-session.v1",
        "session_id": SESSION_ID,
        "designated": True,
        "target_claim_level": 0,
        "claim_level": None,
        "collection_started": True,
        "collection_completed": True,
        "all_cases_passed": True,
        "candidate_binary_sha256": CANDIDATE_SHA256,
        "repository_commit": IMPLEMENTATION_COMMIT,
        "source_bundle_sha256": SOURCE_BUNDLE_SHA256,
        "tt_metal_commit": TT_METAL_COMMIT,
        "device_id": 0,
        "device_count": 1,
        "core_count": 1,
        "case_order": list(CASE_IDS),
        "attempts_per_case": 1,
        "retries": 0,
        "replacement_results": 0,
        "stable_benchmark": False,
        "performance_eligible": False,
    }
    for key, value in expected.items():
        _require(session.get(key) == value, f"H2A session {key} mismatch")
    _require(
        [item.get("case_id") for item in session.get("results", ())] == list(CASE_IDS),
        "H2A session result order changed",
    )
    _require(
        all(item.get("attempt") == 1 and item.get("passed") is True for item in session["results"]),
        "H2A session contains failed or repeated cases",
    )
    for result in session["results"]:
        report = _load_json(root / SESSION_ROOT / result["report"])
        metadata = report.get("candidate_metrics", {}).get("candidate_metadata", {})
        _require(report.get("execution_label") == "hardware", "H2A case is not hardware execution")
        _require(report.get("claim_level") is None, "H2A source report was promoted")
        _require(report.get("stable_benchmark") is False, "H2A source report became stable")
        _require(
            report.get("performance_eligible") is False,
            "H2A source report became performance eligible",
        )
        correctness = report.get("correctness", {})
        _require(correctness.get("passed") is True, "H2A case correctness failed")
        _require(correctness.get("failing_value_count") == 0, "H2A case contains failing values")
        _require(
            correctness.get("nonfinite_value_count") == 0, "H2A case contains nonfinite values"
        )
        _require(metadata.get("device_create_count") == 1, "H2A device create count mismatch")
        _require(metadata.get("device_close_count") == 1, "H2A device close count mismatch")
    return session


def _validate_governance_evidence(root: Path, session: Mapping[str, Any]) -> None:
    preflight = _load_json(root / GOVERNANCE_ROOT / "preflight.json")
    for key, value in {
        "dry_run_passed": True,
        "device_health_checked": True,
        "device_scope_validated": True,
        "hardware_executed": False,
        "session_opened": False,
        "source_tree_clean": True,
        "tt_metal_tree_clean": True,
        "candidate_binary_sha256": CANDIDATE_SHA256,
        "source_bundle_sha256": SOURCE_BUNDLE_SHA256,
    }.items():
        _require(preflight.get(key) == value, f"H2A preflight {key} mismatch")
    health = preflight.get("device_health", {})
    _require(
        health.get("device_id") == 0
        and health.get("dram_status") is True
        and health.get("faults") == "0x0"
        and health.get("throttler") == "0x0"
        and health.get("pcie_width") == "16",
        "H2A preflight device health mismatch",
    )
    cpu = _load_json(root / GOVERNANCE_ROOT / "cpu-reference-validation.txt")
    cpu_correctness = cpu.get("correctness", {})
    _require(cpu.get("execution_label") == "cpu_reference", "H2A CPU reference label mismatch")
    _require(cpu_correctness.get("passed") is True, "H2A CPU reference failed")
    _require(cpu_correctness.get("failing_value_count") == 0, "H2A CPU reference failing values")
    _require(
        cpu_correctness.get("nonfinite_value_count") == 0, "H2A CPU reference nonfinite values"
    )
    stdout = _load_json(root / GOVERNANCE_ROOT / "collector.stdout.txt")
    _require(stdout == session, "H2A collector stdout differs from session manifest")
    _require(
        (root / GOVERNANCE_ROOT / "collector.stderr.txt").read_bytes() == b"",
        "H2A collector stderr is not empty",
    )
    _require(
        (root / GOVERNANCE_ROOT / "collector.exit-code.txt").read_text(encoding="utf-8") == "0\n",
        "H2A collector did not exit zero",
    )
    started = (
        (root / GOVERNANCE_ROOT / "collection-invoked-at.txt").read_text(encoding="utf-8").strip()
    )
    finished = (
        (root / GOVERNANCE_ROOT / "collection-returned-at.txt").read_text(encoding="utf-8").strip()
    )
    _require(started == "2026-07-16T03:22:12Z", "H2A collection start changed")
    _require(finished == "2026-07-16T03:22:59Z", "H2A collection end changed")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HamiltonianLoweringReleaseError(f"invalid H2A JSON: {path}") from exc
    _require(isinstance(payload, dict), f"H2A JSON must be an object: {path}")
    return payload


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _text(value: Mapping[str, Any], key: str) -> str:
    result = value.get(key)
    _require(isinstance(result, str) and bool(result), f"H2A {key} must be nonempty")
    return result


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HamiltonianLoweringReleaseError(message)
