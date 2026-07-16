"""Frozen H2A Claim Level 0 input, collection, and qualification contract."""

from __future__ import annotations

from array import array
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Callable

import torch

from tt_rqm_kernels.hamiltonian_lowering_benchmark import CASE_IDS, reference_cases
from tt_rqm_kernels.hamiltonian_lowering_candidate import (
    CandidateRun,
    HamiltonianLoweringCandidateError,
    run_external_candidate,
)
from tt_rqm_kernels.hamiltonian_lowering_source_identity import (
    validate_source_manifest,
)

DESIGNATED_SCHEMA = "tt-rqm-hamiltonian-lowering-designated-conformance.v1"
INPUT_SCHEMA = "tt-rqm-hamiltonian-lowering-frozen-inputs.v1"
CASE_SCHEMA = "tt-rqm-hamiltonian-lowering-frozen-case.v1"
SESSION_SCHEMA = "tt-rqm-hamiltonian-lowering-designated-session.v1"
QUALIFICATION_SCHEMA = "tt-rqm-hamiltonian-lowering-qualification.v1"
IMPLEMENTATION_COMMIT = "225cb213ae79df7acd43d6056841c3eae7b5fc40"
SOURCE_BUNDLE_SHA256 = "519b2b9ffb7341893aed1574604ce3c0021b9c47830ca9c297d03d69b7cf80d5"
CANDIDATE_SHA256 = "b12063fd8ff73ff7372713eeb3fbdea31c56462c94e314713909a1f07e225979"
TT_METAL_COMMIT = "dd2849b5bc6b7a5d38a9eafbeba31ef8d530f8d4"
COMPILER_IDENTITY = "c++ (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0"
RUNTIME_IDENTITY = f"tt-metal-{TT_METAL_COMMIT}"
MANIFEST_PATH = Path("benchmarks/manifests/hamiltonian-lowering-h2a-designated-conformance.json")
SOURCE_MANIFEST_PATH = Path("benchmarks/manifests/hamiltonian-lowering-h2a-source-manifest.json")
INPUT_ROOT = Path("benchmarks/inputs/hamiltonian-lowering-h2a-designated-conformance")
INPUT_MANIFEST_PATH = INPUT_ROOT / "input-manifest.json"
DEVELOPMENT_IDENTITIES = frozenset(
    {
        "ca24f5253b8869ca92621e6031cc08c1d4bdafe669185e02593671a8727f3792",
        "433e74b827d2cf9a7a790a6c9d7bb3917fc1fed3915ec384de0486cdc014d306",
        "a307055702acd4f370d80ee8fa9a59a48e81f209d3174e9d5358d61e544bdeed",
        "7fb65217e05139bf035952ebeb34602d49e5f1772b8dec4c336b7a296e1fba2f",
    }
)


class HamiltonianLoweringDesignatedError(RuntimeError):
    """Raised when the frozen designated contract is violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze_inputs(output_root: Path) -> dict[str, Any]:
    """Write the nine seed-zero input packages using raw little-endian FP32."""

    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise HamiltonianLoweringDesignatedError("frozen input directory must be new or empty")
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for case in reference_cases(seed=0):
        case_id = case["id"]
        case_root = output_root / case_id
        case_root.mkdir()
        hamiltonians = case["hamiltonians"].detach().cpu().to(torch.float32).contiguous()
        dt = torch.as_tensor(case["dt"], dtype=torch.float32).detach().cpu().contiguous()
        _write_float32(case_root / "hamiltonians.bin", hamiltonians)
        _write_float32(case_root / "dt.bin", dt)
        record = {
            "schema": CASE_SCHEMA,
            "case_id": case_id,
            "B": int(hamiltonians.shape[0]),
            "K": int(hamiltonians.shape[1]),
            "dtype": "float32",
            "hamiltonian_shape": list(hamiltonians.shape),
            "dt_shape": list(dt.shape),
            "hamiltonians_path": f"{case_id}/hamiltonians.bin",
            "dt_path": f"{case_id}/dt.bin",
            "hamiltonians_sha256": sha256_file(case_root / "hamiltonians.bin"),
            "dt_sha256": sha256_file(case_root / "dt.bin"),
            "rotor_shape": [int(hamiltonians.shape[0]), int(hamiltonians.shape[1]), 4],
            "phase_shape": [int(hamiltonians.shape[0]), int(hamiltonians.shape[1]), 2],
        }
        _write_json(case_root / "case.json", record)
        records.append(record)
    manifest = {
        "schema": INPUT_SCHEMA,
        "generator_seed": 0,
        "case_order": list(CASE_IDS),
        "serialization": "raw little-endian float32 row-major",
        "hbar": 1.0,
        "input_lane_order": ["h0", "hx", "hy", "hz"],
        "rotor_lane_order": ["w", "x", "y", "z"],
        "phase_lane_order": ["real", "imag"],
        "cases": records,
    }
    _write_json(output_root / "input-manifest.json", manifest)
    return manifest


def validate_frozen_inputs(input_root: Path, *, compare_generator: bool = True) -> dict[str, Any]:
    input_root = input_root.resolve()
    manifest = _load_json(input_root / "input-manifest.json")
    _require(manifest.get("schema") == INPUT_SCHEMA, "frozen input schema mismatch")
    _require(manifest.get("case_order") == list(CASE_IDS), "frozen case order mismatch")
    records = manifest.get("cases")
    _require(isinstance(records, list), "frozen cases must be a list")
    _require(
        [record.get("case_id") for record in records] == list(CASE_IDS), "case records reordered"
    )
    generated = {case["id"]: case for case in reference_cases(seed=0)}
    for record in records:
        case_id = record["case_id"]
        case_record = _load_json(input_root / case_id / "case.json")
        _require(case_record == record, f"{case_id} case manifest mismatch")
        h_path = input_root / record["hamiltonians_path"]
        dt_path = input_root / record["dt_path"]
        _require(
            sha256_file(h_path) == record["hamiltonians_sha256"],
            f"{case_id} Hamiltonian hash mismatch",
        )
        _require(sha256_file(dt_path) == record["dt_sha256"], f"{case_id} dt hash mismatch")
        if compare_generator:
            case = generated[case_id]
            expected_h = (
                case["hamiltonians"].detach().cpu().to(torch.float32).contiguous().numpy().tobytes()
            )
            expected_dt = (
                torch.as_tensor(case["dt"], dtype=torch.float32).contiguous().numpy().tobytes()
            )
            _require(
                h_path.read_bytes() == expected_h, f"{case_id} Hamiltonians differ from generator"
            )
            _require(dt_path.read_bytes() == expected_dt, f"{case_id} dt differs from generator")
    return {
        "frozen_inputs_valid": True,
        "case_count": len(records),
        "input_manifest_sha256": sha256_file(input_root / "input-manifest.json"),
    }


def validate_designated_manifest(manifest_path: Path, repo_root: Path) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    _require(manifest.get("schema") == DESIGNATED_SCHEMA, "designated manifest schema mismatch")
    expected = {
        "benchmark_id": "hamiltonian-lowering-h2a-designated-conformance",
        "stage": "H2A",
        "target_claim_level": 0,
        "status": "frozen_not_collected",
        "repository_commit": IMPLEMENTATION_COMMIT,
        "source_bundle_sha256": SOURCE_BUNDLE_SHA256,
        "candidate_binary_sha256": CANDIDATE_SHA256,
        "raw_binary_sha256": [CANDIDATE_SHA256, CANDIDATE_SHA256],
        "tt_metal_commit": TT_METAL_COMMIT,
        "compiler_identity": COMPILER_IDENTITY,
        "runtime_identity": RUNTIME_IDENTITY,
        "device_id": 0,
        "device_count": 1,
        "core_count": 1,
        "case_order": list(CASE_IDS),
        "designated_session_count": 1,
        "discard_or_replace_results": "forbidden",
        "claim_level": None,
        "performance_eligible": False,
        "stable_benchmark": False,
    }
    for key, value in expected.items():
        _require(manifest.get(key) == value, f"designated manifest {key} mismatch")
    _require(
        not DEVELOPMENT_IDENTITIES.intersection(_manifest_identity_values(manifest)),
        "development identity reused",
    )
    _require(manifest.get("completed_session_id") is None, "completed session must be absent")
    _require(manifest.get("collection_started") is False, "collection must not be started")
    required_nonclaims = {
        "no_collection_yet",
        "no_claim_level_yet",
        "no_performance_claim",
        "no_stability_claim",
        "no_acceleration_claim",
        "no_bandwidth_claim",
        "no_energy_claim",
        "no_dual_device_claim",
        "no_H2B_claim",
        "no_Tenstorrent_endorsement",
        "no_inheritance_from_H1",
    }
    _require(
        set(manifest.get("nonclaims", ())) == required_nonclaims, "designated nonclaims mismatch"
    )
    source_manifest = repo_root / manifest["source_manifest"]
    source_result = validate_source_manifest(
        source_manifest, repo_root, expected_commit=IMPLEMENTATION_COMMIT
    )
    _require(
        sha256_file(source_manifest) == manifest["source_manifest_sha256"],
        "source manifest hash mismatch",
    )
    input_root = repo_root / manifest["frozen_input_root"]
    input_result = validate_frozen_inputs(input_root)
    _require(
        input_result["input_manifest_sha256"] == manifest["input_manifest_sha256"],
        "input manifest hash mismatch",
    )
    _require(
        manifest.get("cases") == _load_json(input_root / "input-manifest.json")["cases"],
        "case contract mismatch",
    )
    return {"designated_manifest_valid": True, **source_result, **input_result}


def build_designated_manifest(repo_root: Path) -> dict[str, Any]:
    """Build the frozen-not-collected contract from already frozen artifacts."""

    repo_root = repo_root.resolve()
    source_manifest = repo_root / SOURCE_MANIFEST_PATH
    inputs = _load_json(repo_root / INPUT_MANIFEST_PATH)
    return {
        "schema": DESIGNATED_SCHEMA,
        "benchmark_id": "hamiltonian-lowering-h2a-designated-conformance",
        "title": "H2A Compensated Device-Side Hamiltonian Lowering Designated Conformance",
        "stage": "H2A",
        "target_claim_level": 0,
        "status": "frozen_not_collected",
        "repository_commit": IMPLEMENTATION_COMMIT,
        "source_bundle_sha256": SOURCE_BUNDLE_SHA256,
        "source_manifest": SOURCE_MANIFEST_PATH.as_posix(),
        "source_manifest_sha256": sha256_file(source_manifest),
        "candidate_binary_sha256": CANDIDATE_SHA256,
        "raw_binary_sha256": [CANDIDATE_SHA256, CANDIDATE_SHA256],
        "clean_builds_byte_identical": True,
        "tt_metal_commit": TT_METAL_COMMIT,
        "compiler_identity": COMPILER_IDENTITY,
        "runtime_identity": RUNTIME_IDENTITY,
        "device_scope": "one Wormhole N300 device; one Tensix compute core",
        "device_id": 0,
        "device_count": 1,
        "core_count": 1,
        "frozen_input_root": INPUT_ROOT.as_posix(),
        "input_manifest_sha256": sha256_file(repo_root / INPUT_MANIFEST_PATH),
        "case_order": list(CASE_IDS),
        "cases": inputs["cases"],
        "hbar": 1.0,
        "dtype": "float32",
        "input_lane_order": ["h0", "hx", "hy", "hz"],
        "rotor_lane_order": ["w", "x", "y", "z"],
        "phase_lane_order": ["real", "imag"],
        "zero_mask_strategy": "eqz(r2), safe denominator select before reciprocal, identity output select",
        "compensated_product_strategy": "device-side Dekker split TwoProduct with FP32 splitter 4097",
        "period_reduction_strategy": "device-side nearest-multiple split-2pi reduction",
        "tolerances": {
            "rotor_atol": 0.0001,
            "rotor_rtol": 0.0001,
            "phase_atol": 0.0001,
            "phase_rtol": 0.0001,
        },
        "validation": {
            "zero_failing_values_required": True,
            "zero_nonfinite_values_required": True,
            "whole_output_validation_required": True,
            "complex128_matrix_diagnostic_required": True,
        },
        "designated_session_count": 1,
        "discard_or_replace_results": "forbidden",
        "collection_started": False,
        "completed_session_id": None,
        "claim_level": None,
        "performance_eligible": False,
        "stable_benchmark": False,
        "nonclaims": [
            "no_collection_yet",
            "no_claim_level_yet",
            "no_performance_claim",
            "no_stability_claim",
            "no_acceleration_claim",
            "no_bandwidth_claim",
            "no_energy_claim",
            "no_dual_device_claim",
            "no_H2B_claim",
            "no_Tenstorrent_endorsement",
            "no_inheritance_from_H1",
        ],
    }


def dry_run_preflight(
    *,
    manifest_path: Path,
    governance_root: Path,
    source_repo: Path,
    tt_metal_root: Path,
    candidate_binary: Path,
    compiler: str = "c++",
    tt_smi: str = "tt-smi",
) -> dict[str, Any]:
    """Validate all frozen identities without device execution or session creation."""

    manifest_result = validate_designated_manifest(manifest_path, governance_root)
    manifest = _load_json(manifest_path)
    _require(
        _git(source_repo, "rev-parse", "HEAD") == IMPLEMENTATION_COMMIT,
        "source checkout commit mismatch",
    )
    _require(
        not _git(source_repo, "status", "--porcelain", "--untracked-files=all"),
        "source checkout is dirty",
    )
    validate_source_manifest(
        governance_root / manifest["source_manifest"],
        source_repo,
        expected_commit=IMPLEMENTATION_COMMIT,
    )
    _require(
        _git(tt_metal_root, "rev-parse", "HEAD") == TT_METAL_COMMIT, "TT-Metal commit mismatch"
    )
    _require(
        not _git(tt_metal_root, "status", "--porcelain", "--untracked-files=all"),
        "TT-Metal tree is dirty",
    )
    _require(sha256_file(candidate_binary) == CANDIDATE_SHA256, "candidate binary hash mismatch")
    compiler_line = subprocess.run(
        [compiler, "--version"], check=True, capture_output=True, text=True
    ).stdout.splitlines()[0]
    _require(compiler_line == COMPILER_IDENTITY, "compiler identity mismatch")
    device_health = query_device_health(tt_smi)
    return {
        **manifest_result,
        "dry_run_passed": True,
        "source_tree_clean": True,
        "tt_metal_tree_clean": True,
        "candidate_binary_sha256": CANDIDATE_SHA256,
        "device_scope_validated": True,
        "device_health_checked": True,
        "device_health": device_health,
        "hardware_executed": False,
        "session_opened": False,
    }


def query_device_health(tt_smi: str = "tt-smi") -> dict[str, Any]:
    """Read non-executing device telemetry and require healthy N300 device 0."""

    completed = subprocess.run([tt_smi, "-s"], check=True, capture_output=True, text=True)
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise HamiltonianLoweringDesignatedError("tt-smi health JSON is malformed") from exc
    return validate_device_health(payload)


def validate_device_health(payload: dict[str, Any]) -> dict[str, Any]:
    devices = payload.get("device_info")
    _require(isinstance(devices, list) and bool(devices), "tt-smi found no devices")
    device = devices[0]
    _require(isinstance(device, dict), "tt-smi device 0 record is malformed")
    board = device.get("board_info", {})
    telemetry = device.get("smbus_telem", {})
    _require("n300" in str(board.get("board_type", "")).lower(), "device 0 is not N300")
    _require(board.get("dram_status") is True, "device 0 DRAM is unhealthy")
    _require(str(telemetry.get("FAULTS", "")).lower() == "0x0", "device 0 reports faults")
    _require(str(telemetry.get("THROTTLER", "")).lower() == "0x0", "device 0 is throttled")
    _require(board.get("pcie_width") == "16", "device 0 PCIe width mismatch")
    return {
        "device_id": 0,
        "board_type": board["board_type"],
        "dram_status": True,
        "pcie_width": board["pcie_width"],
        "faults": "0x0",
        "throttler": "0x0",
    }


def collect_designated_session(
    *,
    manifest_path: Path,
    governance_root: Path,
    source_repo: Path,
    tt_metal_root: Path,
    candidate_binary: Path,
    output_root: Path,
    session_id: str,
    runner: Callable[..., CandidateRun] = run_external_candidate,
    tt_smi: str = "tt-smi",
) -> dict[str, Any]:
    """Open exactly one session and retain one attempt for every frozen case."""

    dry_run_preflight(
        manifest_path=manifest_path,
        governance_root=governance_root,
        source_repo=source_repo,
        tt_metal_root=tt_metal_root,
        candidate_binary=candidate_binary,
        tt_smi=tt_smi,
    )
    _require(session_id.strip() == session_id and bool(session_id), "session id is invalid")
    _require(not output_root.exists(), "designated session output already exists")
    output_root.mkdir(parents=True)
    cases_root = output_root / "cases"
    cases_root.mkdir()
    manifest = _load_json(manifest_path)
    session = {
        "schema": SESSION_SCHEMA,
        "session_id": session_id,
        "designated": True,
        "target_claim_level": 0,
        "claim_level": None,
        "collection_started": True,
        "collection_completed": False,
        "manifest_sha256": sha256_file(manifest_path),
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
        "results": [],
    }
    _write_json(output_root / "session-manifest.json", session)
    runner_path = (
        source_repo.resolve()
        / "experimental/tt_metalium_hamiltonian_lowering_compensated/run_candidate.py"
    )
    command = " ".join(
        shlex.quote(value)
        for value in (
            "env",
            f"TT_METAL_HOME={tt_metal_root.resolve()}",
            f"TT_METAL_RUNTIME_ROOT={tt_metal_root.resolve()}",
            f"TT_RQM_H2A_BINARY={candidate_binary.resolve()}",
            sys.executable,
            str(runner_path),
        )
    )
    input_manifest = _load_json(
        governance_root / manifest["frozen_input_root"] / "input-manifest.json"
    )
    for record in input_manifest["cases"]:
        case_id = record["case_id"]
        case_root = cases_root / case_id
        case_root.mkdir()
        result = {"case_id": case_id, "attempt": 1, "passed": False}
        try:
            hamiltonians, dt = load_frozen_case(
                governance_root / manifest["frozen_input_root"], record
            )
            run = runner(hamiltonians, dt, command=command, execution_label="hardware")
            _retain_run(case_root, run)
            result.update(
                {
                    "passed": True,
                    "report": f"cases/{case_id}/report.json",
                    "checksum": run.report["correctness"]["checksum"],
                }
            )
        except (
            HamiltonianLoweringCandidateError,
            HamiltonianLoweringDesignatedError,
            OSError,
        ) as exc:
            (case_root / "error.txt").write_text(str(exc) + "\n", encoding="utf-8")
            result["error"] = str(exc)
        session["results"].append(result)
        _write_json(output_root / "session-manifest.json", session)
    session["collection_completed"] = True
    session["all_cases_passed"] = all(item["passed"] for item in session["results"])
    _write_json(output_root / "session-manifest.json", session)
    return session


def qualify_session(session_root: Path, manifest_path: Path, repo_root: Path) -> dict[str, Any]:
    """Deterministically qualify retained evidence without publishing a release."""

    validate_designated_manifest(manifest_path, repo_root)
    frozen = _load_json(manifest_path)
    session = _load_json(session_root / "session-manifest.json")
    _require(session.get("schema") == SESSION_SCHEMA, "session schema mismatch")
    expected = {
        "designated": True,
        "target_claim_level": 0,
        "claim_level": None,
        "collection_started": True,
        "collection_completed": True,
        "manifest_sha256": sha256_file(manifest_path),
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
        _require(session.get(key) == value, f"session {key} mismatch")
    results = session.get("results")
    _require(
        isinstance(results, list) and [item.get("case_id") for item in results] == list(CASE_IDS),
        "session results reordered",
    )
    input_records = {item["case_id"]: item for item in frozen["cases"]}
    for item in results:
        _require(
            item.get("attempt") == 1 and item.get("passed") is True,
            f"case {item.get('case_id')} failed",
        )
        report = _load_json(session_root / item["report"])
        case_id = item["case_id"]
        record = input_records[case_id]
        _require(report.get("execution_label") == "hardware", f"{case_id} is not hardware")
        _require(
            report.get("input_hashes")
            == {
                "hamiltonians_sha256": record["hamiltonians_sha256"],
                "dt_sha256": record["dt_sha256"],
            },
            f"{case_id} inputs mismatch",
        )
        correctness = report.get("correctness", {})
        _require(correctness.get("passed") is True, f"{case_id} correctness failed")
        _require(
            correctness.get("failing_value_count") == 0
            and correctness.get("nonfinite_value_count") == 0,
            f"{case_id} output gate failed",
        )
        metadata = report.get("candidate_metrics", {}).get("candidate_metadata", {})
        for key, value in {
            "candidate_sha256": CANDIDATE_SHA256,
            "source_commit": IMPLEMENTATION_COMMIT,
            "source_tree_clean": True,
            "source_bundle_sha256": SOURCE_BUNDLE_SHA256,
            "tt_metal_commit": TT_METAL_COMMIT,
            "compiler_version": COMPILER_IDENTITY,
            "runtime_version": RUNTIME_IDENTITY,
            "device_id": 0,
            "device_count": 1,
            "core_count": 1,
        }.items():
            _require(metadata.get(key) == value, f"{case_id} metadata {key} mismatch")
        payload = (session_root / "cases" / case_id / "rotors.bin").read_bytes() + (
            session_root / "cases" / case_id / "phases.bin"
        ).read_bytes()
        _require(
            hashlib.sha256(payload).hexdigest() == item["checksum"] == correctness["checksum"],
            f"{case_id} checksum mismatch",
        )
    return {
        "schema": QUALIFICATION_SCHEMA,
        "qualification_passed": True,
        "target_claim_level": 0,
        "claim_level": None,
        "stable_benchmark": False,
        "performance_eligible": False,
        "release_created": False,
    }


def load_frozen_case(input_root: Path, record: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    hamiltonians = _read_float32(
        input_root / record["hamiltonians_path"], tuple(record["hamiltonian_shape"])
    )
    dt = _read_float32(input_root / record["dt_path"], tuple(record["dt_shape"]))
    return hamiltonians, dt


def contract_readiness(manifest_path: Path, repo_root: Path) -> dict[str, Any]:
    result = validate_designated_manifest(manifest_path, repo_root)
    return {
        **result,
        "qualifier_ready": True,
        "designated_session_present": False,
        "qualification_passed": None,
        "claim_level": None,
    }


def _retain_run(case_root: Path, run: CandidateRun) -> None:
    _write_json(case_root / "report.json", run.report)
    (case_root / "stdout.txt").write_text(run.stdout, encoding="utf-8")
    (case_root / "stderr.txt").write_text(run.stderr, encoding="utf-8")
    _write_float32(case_root / "rotors.bin", run.rotors)
    _write_float32(case_root / "phases.bin", run.phases)


def _manifest_identity_values(manifest: dict[str, Any]) -> set[str]:
    values = {
        str(manifest.get(key, ""))
        for key in ("repository_commit", "source_bundle_sha256", "candidate_binary_sha256")
    }
    values.update(str(value) for value in manifest.get("raw_binary_sha256", ()))
    return values


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HamiltonianLoweringDesignatedError(message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise HamiltonianLoweringDesignatedError(f"invalid or missing {path}") from exc
    _require(isinstance(value, dict), f"{path} must contain an object")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_float32(path: Path, value: torch.Tensor) -> None:
    payload = array("f", value.detach().cpu().contiguous().reshape(-1).tolist())
    if sys.byteorder != "little":
        payload.byteswap()
    path.write_bytes(payload.tobytes())


def _read_float32(path: Path, shape: tuple[int, ...]) -> torch.Tensor:
    values = array("f")
    values.frombytes(path.read_bytes())
    if sys.byteorder != "little":
        values.byteswap()
    return torch.tensor(values, dtype=torch.float32).reshape(shape)
