"""Non-designated H2A hardware-pilot collection and offline validation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
from typing import Any

import torch

from tt_rqm_kernels.hamiltonian_lowering_benchmark import CASE_IDS, reference_cases
from tt_rqm_kernels.hamiltonian_lowering_candidate import (
    HamiltonianLoweringCandidateError,
    run_external_candidate,
)

PILOT_SCHEMA = "tt-rqm-hamiltonian-lowering-pilot.v1"
SUITE_SCHEMA = "tt-rqm-hamiltonian-lowering-pilot-suite.v1"
PINNED_TT_METAL_COMMIT = "dd2849b5bc6b7a5d38a9eafbeba31ef8d530f8d4"


class HamiltonianLoweringPilotError(RuntimeError):
    """Raised when retained H2A pilot evidence violates its frozen contract."""


def frozen_case_input_hashes() -> dict[str, dict[str, str]]:
    """Return raw protocol hashes for the exact seed-zero semantic cases."""

    return {
        case["id"]: {
            "hamiltonians_sha256": _tensor_bytes_sha256(case["hamiltonians"]),
            "dt_sha256": _tensor_bytes_sha256(torch.as_tensor(case["dt"], dtype=torch.float32)),
        }
        for case in reference_cases(seed=0)
    }


def collect_pilot(*, command: str, output_dir: Path, pilot_id: str) -> dict[str, Any]:
    """Run every frozen case once; retain all failures without replacement."""

    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise HamiltonianLoweringPilotError("pilot output directory must be new or empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_dir = output_dir / "cases"
    cases_dir.mkdir()
    manifest = {
        "schema": PILOT_SCHEMA,
        "pilot_id": pilot_id,
        "benchmark_family": "HamiltonianLoweringBench",
        "pilot_started": True,
        "designated": False,
        "qualification_eligible": False,
        "claim_level": None,
        "stable_benchmark": False,
        "performance_eligible": False,
        "hardware_execution": True,
        "case_ids": list(CASE_IDS),
        "case_input_hashes": frozen_case_input_hashes(),
        "tt_metal_commit": PINNED_TT_METAL_COMMIT,
    }
    _write_json(output_dir / "pilot-manifest.json", manifest)
    results: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    metadata_records: list[dict[str, Any]] = []
    for case in reference_cases(seed=0):
        case_dir = cases_dir / case["id"]
        case_dir.mkdir()
        entry: dict[str, Any] = {"case_id": case["id"], "passed": False}
        try:
            run = run_external_candidate(
                case["hamiltonians"],
                case["dt"],
                command=command,
                execution_label="hardware",
            )
            _write_json(case_dir / "report.json", run.report)
            (case_dir / "stdout.txt").write_text(run.stdout, encoding="utf-8")
            (case_dir / "stderr.txt").write_text(run.stderr, encoding="utf-8")
            (case_dir / "rotors.bin").write_bytes(run.rotors.contiguous().numpy().tobytes())
            (case_dir / "phases.bin").write_bytes(run.phases.contiguous().numpy().tobytes())
            metadata = run.report["candidate_metrics"]["candidate_metadata"]
            metadata_records.append(metadata)
            identities.append(_identity(metadata))
            entry.update(
                {
                    "passed": True,
                    "report": f"cases/{case['id']}/report.json",
                    "output_checksum": run.report["correctness"]["checksum"],
                    "correctness": run.report["correctness"],
                }
            )
        except HamiltonianLoweringCandidateError as exc:
            message = str(exc)
            (case_dir / "error.txt").write_text(message + "\n", encoding="utf-8")
            entry["error"] = message
        results.append(entry)
    suite_passed = all(result["passed"] for result in results) and bool(identities)
    identity_consistent = bool(identities) and all(value == identities[0] for value in identities)
    suite = {
        "schema": SUITE_SCHEMA,
        "pilot_id": pilot_id,
        "case_ids": list(CASE_IDS),
        "results": results,
        "candidate_identity_consistent": identity_consistent,
        "candidate_identity": identities[0] if identity_consistent else None,
        "suite_passed": suite_passed and identity_consistent,
        "stable_benchmark": False,
        "performance_eligible": False,
        "claim_level": None,
    }
    if identity_consistent:
        _write_json(output_dir / "candidate-metadata.json", metadata_records[0])
    (output_dir / "environment.txt").write_text(
        "\n".join(
            (
                f"host={platform.node()}",
                f"platform={platform.platform()}",
                f"python={platform.python_version()}",
                f"command={command}",
                f"tt_metal_home={os.environ.get('TT_METAL_HOME', '')}",
                "designated=false",
                "qualification_eligible=false",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(output_dir / "suite-report.json", suite)
    (output_dir / "suite-report.md").write_text(_suite_markdown(suite), encoding="utf-8")
    return suite


def validate_pilot_package(root: Path) -> dict[str, Any]:
    """Validate retained evidence without requiring Tenstorrent hardware."""

    root = root.resolve()
    manifest = _load_json(root / "pilot-manifest.json")
    suite = _load_json(root / "suite-report.json")
    if manifest.get("schema") != PILOT_SCHEMA or suite.get("schema") != SUITE_SCHEMA:
        raise HamiltonianLoweringPilotError("pilot schema mismatch")
    for key, expected in {
        "designated": False,
        "qualification_eligible": False,
        "claim_level": None,
        "stable_benchmark": False,
        "performance_eligible": False,
        "hardware_execution": True,
    }.items():
        if manifest.get(key) != expected:
            raise HamiltonianLoweringPilotError(f"pilot manifest {key} must be {expected!r}")
    if manifest.get("tt_metal_commit") != PINNED_TT_METAL_COMMIT:
        raise HamiltonianLoweringPilotError("pilot TT-Metal commit mismatch")
    expected_ids = list(CASE_IDS)
    if manifest.get("case_ids") != expected_ids or suite.get("case_ids") != expected_ids:
        raise HamiltonianLoweringPilotError("pilot case order mismatch")
    if manifest.get("case_input_hashes") != frozen_case_input_hashes():
        raise HamiltonianLoweringPilotError("pilot frozen input hashes changed")
    results = suite.get("results")
    if not isinstance(results, list) or [item.get("case_id") for item in results] != expected_ids:
        raise HamiltonianLoweringPilotError("pilot results are missing or reordered")
    identities: list[dict[str, Any]] = []
    for item in results:
        case_id = item["case_id"]
        if not item.get("passed"):
            if not isinstance(item.get("error"), str) or not item["error"]:
                raise HamiltonianLoweringPilotError(f"failed case {case_id} lacks retained error")
            continue
        report = _load_json(root / item["report"])
        if report.get("execution_label") != "hardware":
            raise HamiltonianLoweringPilotError(f"case {case_id} is not hardware execution")
        if report.get("input_hashes") != manifest["case_input_hashes"][case_id]:
            raise HamiltonianLoweringPilotError(f"case {case_id} input hash mismatch")
        correctness = report.get("correctness", {})
        if (
            correctness.get("passed") is not True
            or correctness.get("failing_value_count") != 0
            or correctness.get("nonfinite_value_count") != 0
        ):
            raise HamiltonianLoweringPilotError(f"case {case_id} correctness mismatch")
        payload = (root / "cases" / case_id / "rotors.bin").read_bytes() + (
            root / "cases" / case_id / "phases.bin"
        ).read_bytes()
        checksum = hashlib.sha256(payload).hexdigest()
        if checksum != correctness.get("checksum") or checksum != item.get("output_checksum"):
            raise HamiltonianLoweringPilotError(f"case {case_id} output checksum mismatch")
        metadata = report.get("candidate_metrics", {}).get("candidate_metadata", {})
        identities.append(_identity(metadata))
    consistent = bool(identities) and all(value == identities[0] for value in identities)
    if identities and not consistent:
        raise HamiltonianLoweringPilotError("candidate identity changed between pilot cases")
    expected_passed = all(item.get("passed") is True for item in results) and consistent
    if expected_passed:
        package_metadata = _load_json(root / "candidate-metadata.json")
        if _identity(package_metadata) != identities[0]:
            raise HamiltonianLoweringPilotError("package candidate metadata mismatch")
    if suite.get("suite_passed") is not expected_passed:
        raise HamiltonianLoweringPilotError("suite_passed is inconsistent with retained cases")
    if suite.get("stable_benchmark") is not False or suite.get("performance_eligible") is not False:
        raise HamiltonianLoweringPilotError("pilot suite cannot be stable or performance eligible")
    if suite.get("claim_level") is not None:
        raise HamiltonianLoweringPilotError("pilot suite cannot invent a claim level")
    return {"package_valid": True, "pilot_passed": expected_passed, "case_count": len(results)}


def _identity(metadata: dict[str, Any]) -> dict[str, Any]:
    identity = {
        key: metadata.get(key)
        for key in (
            "candidate_sha256",
            "source_bundle_sha256",
            "source_commit",
            "tt_metal_commit",
            "device_arch",
            "device_id",
            "core_count",
        )
    }
    if identity["tt_metal_commit"] != PINNED_TT_METAL_COMMIT:
        raise HamiltonianLoweringPilotError("case TT-Metal identity mismatch")
    if identity["device_id"] != 0 or identity["core_count"] != 1:
        raise HamiltonianLoweringPilotError("pilot must use device 0 and one Tensix core")
    if "wormhole" not in str(identity["device_arch"]).lower():
        raise HamiltonianLoweringPilotError("pilot device must be Wormhole")
    for key in ("candidate_sha256", "source_bundle_sha256"):
        value = identity[key]
        if not isinstance(value, str) or len(value) != 64:
            raise HamiltonianLoweringPilotError(f"invalid {key}")
    return identity


def _tensor_bytes_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise HamiltonianLoweringPilotError(f"invalid or missing {path.name}") from exc
    if not isinstance(payload, dict):
        raise HamiltonianLoweringPilotError(f"{path.name} must contain an object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _suite_markdown(suite: dict[str, Any]) -> str:
    lines = ["# H2A non-designated hardware pilot", "", f"Pilot: `{suite['pilot_id']}`", ""]
    for result in suite["results"]:
        lines.append(f"- `{result['case_id']}`: {'pass' if result['passed'] else 'fail'}")
    lines.extend(
        [
            "",
            f"Overall pass: `{str(suite['suite_passed']).lower()}`",
            "",
            "This package is non-designated, qualification-ineligible, and establishes no claim level.",
        ]
    )
    return "\n".join(lines) + "\n"
