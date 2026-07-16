"""Frozen non-designated H2B N300 pilot contract and deterministic inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from tt_rqm_kernels.hamiltonian_evolution_benchmark import reference_cases
from tt_rqm_kernels.hamiltonian_evolution_candidate import PROTOCOL, TT_METAL_COMMIT
from tt_rqm_kernels.hamiltonian_evolution_diagnostics import MATRIX_THRESHOLD
from tt_rqm_kernels.hamiltonian_evolution_domain import (
    PHASE_ANGLE_LIMIT,
    ROTOR_ANGLE_LIMIT,
    angle_extrema,
    validate_pilot_domain,
)
from tt_rqm_kernels.hamiltonian_evolution_source_identity import validate_source_manifest

SCHEMA = "tt-rqm-hamiltonian-evolution-pilot-preregistration.v1"
INPUT_MANIFEST_SCHEMA = "tt-rqm-hamiltonian-evolution-pilot-inputs.v1"
CASE_SCHEMA = "tt-rqm-hamiltonian-evolution-pilot-case.v1"
DEFAULT_MANIFEST = Path("benchmarks/manifests/hamiltonian-evolution-h2b-pilot-v1.json")
DEFAULT_INPUT_ROOT = Path("benchmarks/inputs/hamiltonian-evolution-h2b-pilot-v1")
SOURCE_MANIFEST = Path("benchmarks/manifests/hamiltonian-evolution-h2b-source-manifest.json")
BUILD_REPORT = Path("reports/h2b_clean_build_reproduction.json")
ATOL = 1e-4
RTOL = 1e-4

CASE_ORDER = (
    "identity_k1",
    "zero_vector_phase_chain",
    "axis_x",
    "axis_y",
    "axis_z",
    "noncommuting_xy",
    "noncommuting_yx",
    "mixed_zero_nonzero",
    "tiny_norms",
    "varying_dt",
    "random_finite",
    "long_chain",
    "boundary_rotor_positive",
    "boundary_rotor_negative",
    "boundary_phase_positive",
    "boundary_phase_negative",
    "boundary_combined",
    "boundary_noncommuting_xy",
    "boundary_noncommuting_yx",
    "large_angle_short_chain",
)
STRESS_CASES = frozenset({"large_angle_short_chain"})


class HamiltonianEvolutionPilotContractError(ValueError):
    """Raised when the H2B pilot contract changes or is incomplete."""


def pilot_cases() -> list[dict[str, Any]]:
    ordinary = {case["id"]: case for case in reference_cases(0)}
    mixed_direction = torch.tensor((2.0, -3.0, 4.0), dtype=torch.float32)
    mixed_direction = mixed_direction / torch.linalg.vector_norm(mixed_direction)
    combined = torch.zeros((1, 1, 4), dtype=torch.float32)
    combined[..., 0] = 8000.0
    combined[..., 1:] = 1000.0 * mixed_direction
    near_x = torch.tensor([[[8000.0, 1000.0, 0.0, 0.0], [-8000.0, 0.0, 1000.0, 0.0]]])
    boundary = {
        "boundary_rotor_positive": {
            "id": "boundary_rotor_positive",
            "hamiltonians": torch.tensor([[[0.0, ROTOR_ANGLE_LIMIT, 0.0, 0.0]]]),
            "dt": 1.0,
        },
        "boundary_rotor_negative": {
            "id": "boundary_rotor_negative",
            "hamiltonians": torch.tensor([[[0.0, ROTOR_ANGLE_LIMIT, 0.0, 0.0]]]),
            "dt": -1.0,
        },
        "boundary_phase_positive": {
            "id": "boundary_phase_positive",
            "hamiltonians": torch.tensor([[[PHASE_ANGLE_LIMIT, 0.0, 0.0, 0.0]]]),
            "dt": 1.0,
        },
        "boundary_phase_negative": {
            "id": "boundary_phase_negative",
            "hamiltonians": torch.tensor([[[-PHASE_ANGLE_LIMIT, 0.0, 0.0, 0.0]]]),
            "dt": 1.0,
        },
        "boundary_combined": {
            "id": "boundary_combined",
            "hamiltonians": combined,
            "dt": 1.0,
        },
        "boundary_noncommuting_xy": {
            "id": "boundary_noncommuting_xy",
            "hamiltonians": near_x,
            "dt": 1.0,
        },
        "boundary_noncommuting_yx": {
            "id": "boundary_noncommuting_yx",
            "hamiltonians": near_x.flip(1).clone(),
            "dt": 1.0,
        },
    }
    combined_cases = ordinary | boundary
    return [
        {
            **combined_cases[case_id],
            "role": "stress_diagnostic" if case_id in STRESS_CASES else "conformance",
            "hbar": 1.0,
        }
        for case_id in CASE_ORDER
    ]


def write_frozen_inputs(root: Path) -> dict[str, Any]:
    root = root.resolve()
    if root.exists() and any(root.iterdir()):
        raise HamiltonianEvolutionPilotContractError("frozen input directory must be new or empty")
    root.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for case in pilot_cases():
        case_dir = root / case["id"]
        case_dir.mkdir()
        hamiltonians = case["hamiltonians"].detach().cpu().to(torch.float32).contiguous()
        dt = torch.as_tensor(case["dt"], dtype=torch.float32).contiguous()
        hamiltonian_path = case_dir / "hamiltonians.bin"
        dt_path = case_dir / "dt.bin"
        hamiltonian_path.write_bytes(hamiltonians.numpy().tobytes())
        dt_path.write_bytes(dt.numpy().tobytes())
        extrema = angle_extrema(hamiltonians, dt, hbar=case["hbar"])
        if case["role"] == "conformance":
            validate_pilot_domain(hamiltonians, dt, hbar=case["hbar"])
        entry = {
            "schema": CASE_SCHEMA,
            "case_id": case["id"],
            "role": case["role"],
            "B": int(hamiltonians.shape[0]),
            "K": int(hamiltonians.shape[1]),
            "hamiltonian_shape": list(hamiltonians.shape),
            "dt_shape": list(dt.shape),
            "hbar": case["hbar"],
            "dtype": "float32",
            "hamiltonians_path": f"{case['id']}/hamiltonians.bin",
            "dt_path": f"{case['id']}/dt.bin",
            "hamiltonians_sha256": _sha256(hamiltonian_path),
            "dt_sha256": _sha256(dt_path),
            "angle_extrema": extrema,
        }
        entries.append(entry)
        (case_dir / "case.json").write_text(
            json.dumps(entry, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    manifest = {
        "schema": INPUT_MANIFEST_SCHEMA,
        "generator": "hamiltonian_evolution_pilot_contract.pilot_cases.v1",
        "seed": 0,
        "case_order": list(CASE_ORDER),
        "cases": entries,
    }
    (root / "input-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def build_pilot_contract(
    repo_root: Path,
    *,
    candidate_binary_sha256: str,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    _require_sha256(candidate_binary_sha256, "candidate_binary_sha256")
    source_path = repo_root / SOURCE_MANIFEST
    source = validate_source_manifest(source_path, repo_root)
    inputs = _load_json(repo_root / DEFAULT_INPUT_ROOT / "input-manifest.json")
    build = _load_json(repo_root / BUILD_REPORT)
    if build.get("byte_identical") is not True:
        raise HamiltonianEvolutionPilotContractError("clean builds are not byte-identical")
    if build.get("candidate_binary_sha256") != candidate_binary_sha256:
        raise HamiltonianEvolutionPilotContractError("build report candidate hash mismatch")
    return {
        "schema": SCHEMA,
        "benchmark_id": "hamiltonian-evolution-h2b-n300-pilot-v1",
        "status": "pilot_frozen_before_first_n300_run",
        "non_designated": True,
        "claim_level": None,
        "stable_benchmark": False,
        "performance_eligible": False,
        "source_commit": source["repository_commit"],
        "source_scope_clean": True,
        "source_manifest": SOURCE_MANIFEST.as_posix(),
        "source_manifest_sha256": _sha256(source_path),
        "source_bundle_sha256": source["source_bundle_sha256"],
        "candidate_binary_sha256": candidate_binary_sha256,
        "build_reproduction_report": BUILD_REPORT.as_posix(),
        "build_reproduction_sha256": _sha256(repo_root / BUILD_REPORT),
        "clean_builds_byte_identical": True,
        "tt_metal_commit": TT_METAL_COMMIT,
        "candidate_protocol": PROTOCOL,
        "seed": 0,
        "dtype": "float32",
        "hbar": 1.0,
        "input_lane_order": ["h0", "hx", "hy", "hz"],
        "final_rotor_lane_order": ["w", "x", "y", "z"],
        "final_phase_lane_order": ["real", "imag"],
        "logical_order": "row-major",
        "frozen_input_root": DEFAULT_INPUT_ROOT.as_posix(),
        "input_manifest_sha256": _sha256(repo_root / DEFAULT_INPUT_ROOT / "input-manifest.json"),
        "case_order": list(CASE_ORDER),
        "cases": inputs["cases"],
        "numerical_domain": {
            "rotor_angle_limit": ROTOR_ANGLE_LIMIT,
            "phase_angle_limit": PHASE_ANGLE_LIMIT,
            "large_angle_short_chain_role": "out_of_domain_stress_diagnostic",
            "public_api_valid_outside_domain": True,
        },
        "tolerances": {
            "final_rotor_atol": ATOL,
            "final_rotor_rtol": RTOL,
            "final_phase_atol": ATOL,
            "final_phase_rtol": RTOL,
            "final_matrix_max_absolute_error": MATRIX_THRESHOLD,
            "failing_values": 0,
            "nonfinite_values": 0,
        },
        "device_contract": {
            "device_count": 1,
            "device_id": 0,
            "device_create_count": 1,
            "device_close_count": 1,
            "program_count": 2,
            "intermediate_storage": "device_dram",
            "device_resident_intermediate": True,
            "intermediate_d2h_count": 0,
            "intermediate_h2d_count": 0,
            "host_round_trip_count": 0,
            "host_intermediate_repacking": False,
            "automatic_normalization": False,
            "composition_order": "K-1 ... 0",
        },
        "attempt_policy": {
            "attempts_per_case": 1,
            "retries": 0,
            "replacement": "forbidden",
            "first_result_retained_regardless_of_outcome": True,
        },
        "required_artifacts": [
            "pilot-manifest.json",
            "contract.json",
            "source-manifest.json",
            "build-reproduction.json",
            "environment.json",
            "pre-run-device-health.json",
            "post-run-device-health.json",
            "cases/<case_id>/manifest.json",
            "cases/<case_id>/hamiltonians.bin",
            "cases/<case_id>/dt.bin",
            "cases/<case_id>/stdout.txt",
            "cases/<case_id>/stderr.txt",
            "cases/<case_id>/exit-status.json",
            "cases/<case_id>/metrics.json when produced",
            "cases/<case_id>/final_rotors.bin when produced",
            "cases/<case_id>/final_phases.bin when produced",
            "cases/<case_id>/report.json or error.json",
            "suite-report.json",
        ],
        "nonclaims": [
            "no_claim_level",
            "no_stability_claim",
            "no_performance_claim",
            "no_acceleration_claim",
            "no_cpu_speedup_claim",
            "no_energy_claim",
            "no_bandwidth_claim",
            "no_application_speedup_claim",
            "no_dual_device_claim",
            "no_h1_inheritance",
            "no_h2a_inheritance",
            "no_tenstorrent_endorsement",
        ],
    }


def validate_pilot_contract(path: Path, repo_root: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if payload.get("schema") != SCHEMA:
        raise HamiltonianEvolutionPilotContractError("pilot contract schema mismatch")
    expected = build_pilot_contract(
        repo_root, candidate_binary_sha256=str(payload.get("candidate_binary_sha256", ""))
    )
    if payload != expected:
        raise HamiltonianEvolutionPilotContractError("pilot contract differs from frozen inputs")
    roles = {case["case_id"]: case["role"] for case in payload["cases"]}
    if roles.get("large_angle_short_chain") != "stress_diagnostic":
        raise HamiltonianEvolutionPilotContractError("large-angle stress role changed")
    if [case["case_id"] for case in payload["cases"]] != list(CASE_ORDER):
        raise HamiltonianEvolutionPilotContractError("pilot case order changed")
    return payload


def write_pilot_contract(path: Path, repo_root: Path, candidate_sha256: str) -> dict[str, Any]:
    payload = build_pilot_contract(repo_root, candidate_binary_sha256=candidate_sha256)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def load_frozen_case(
    repo_root: Path, case_id: str
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    manifest = _load_json(repo_root / DEFAULT_INPUT_ROOT / "input-manifest.json")
    entry = next((case for case in manifest["cases"] if case["case_id"] == case_id), None)
    if entry is None:
        raise HamiltonianEvolutionPilotContractError(f"unknown frozen case: {case_id}")
    hamiltonian_path = repo_root / DEFAULT_INPUT_ROOT / entry["hamiltonians_path"]
    dt_path = repo_root / DEFAULT_INPUT_ROOT / entry["dt_path"]
    if _sha256(hamiltonian_path) != entry["hamiltonians_sha256"]:
        raise HamiltonianEvolutionPilotContractError("Hamiltonian input hash mismatch")
    if _sha256(dt_path) != entry["dt_sha256"]:
        raise HamiltonianEvolutionPilotContractError("dt input hash mismatch")
    hamiltonians = (
        torch.frombuffer(bytearray(hamiltonian_path.read_bytes()), dtype=torch.float32)
        .clone()
        .reshape(entry["hamiltonian_shape"])
    )
    dt = (
        torch.frombuffer(bytearray(dt_path.read_bytes()), dtype=torch.float32)
        .clone()
        .reshape(entry["dt_shape"])
    )
    return hamiltonians, dt, entry


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise HamiltonianEvolutionPilotContractError(f"invalid or missing {path}") from exc
    if not isinstance(payload, dict):
        raise HamiltonianEvolutionPilotContractError(f"{path} must contain an object")
    return payload


def _require_sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise HamiltonianEvolutionPilotContractError(f"{name} must be lowercase SHA-256")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
