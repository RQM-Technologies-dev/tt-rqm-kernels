"""Fail-closed external-candidate protocol for complete H2B evolution."""

from __future__ import annotations

from array import array
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time
from typing import Any, Literal

import torch

from tt_rqm_kernels.hamiltonian.su2_evolution import evolve_two_level_hamiltonian
from tt_rqm_kernels.hamiltonian.su2_reference import (
    compose_hamiltonian_matrices,
    u2_matrix_from_rotor_phase,
)
from tt_rqm_kernels.hamiltonian_evolution_benchmark import ATOL, RTOL
from tt_rqm_kernels.hamiltonian_evolution_domain import validate_pilot_domain

PROTOCOL = "tt-rqm-external-hamiltonian-evolution.v1"
METRICS_SCHEMA = "tt-rqm-external-hamiltonian-evolution-metrics.v1"
REPORT_SCHEMA = "tt-rqm-hamiltonian-evolution-candidate-report.v1"
TT_METAL_COMMIT = "dd2849b5bc6b7a5d38a9eafbeba31ef8d530f8d4"
HARDWARE_ATOL = 1e-4
HARDWARE_RTOL = 1e-4
Stage = Literal["conformance", "performance"]
ExecutionLabel = Literal["cpu_reference", "hardware"]


class HamiltonianEvolutionCandidateError(RuntimeError):
    """Raised when an H2B candidate violates its serialized contract."""


@dataclass(frozen=True)
class CandidateRun:
    report: dict[str, Any]
    final_rotors: torch.Tensor
    final_phases: torch.Tensor
    stdout: str = ""
    stderr: str = ""


def run_external_candidate(
    hamiltonians: torch.Tensor,
    dt: float | torch.Tensor,
    *,
    command: str,
    stage: Stage = "conformance",
    execution_label: ExecutionLabel = "cpu_reference",
    hbar: float = 1.0,
    enforce_pilot_domain: bool = False,
    retained_work_dir: Path | None = None,
) -> CandidateRun:
    """Execute and validate one serialized H2B candidate invocation."""

    if stage not in {"conformance", "performance"}:
        raise HamiltonianEvolutionCandidateError(f"unsupported stage: {stage}")
    if execution_label not in {"cpu_reference", "hardware"}:
        raise HamiltonianEvolutionCandidateError(f"unsupported execution label: {execution_label}")
    if not command or not shlex.split(command):
        raise HamiltonianEvolutionCandidateError("candidate command must not be empty")
    try:
        evolve_two_level_hamiltonian(hamiltonians, dt, hbar=hbar)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise HamiltonianEvolutionCandidateError(str(exc)) from exc
    if hamiltonians.dtype != torch.float32:
        raise HamiltonianEvolutionCandidateError("H2B external inputs must use float32")
    if enforce_pilot_domain:
        try:
            validate_pilot_domain(hamiltonians, dt, hbar=hbar)
        except ValueError as exc:
            raise HamiltonianEvolutionCandidateError(str(exc)) from exc
    coefficients = hamiltonians.detach().cpu().contiguous()
    dt_value = torch.as_tensor(dt, dtype=torch.float32).detach().cpu().contiguous()
    shape = tuple(int(value) for value in coefficients.shape)
    if tuple(dt_value.shape) not in {(), shape[:2]}:
        raise HamiltonianEvolutionCandidateError("H2B external dt must be scalar or exactly [B, K]")

    context = (
        tempfile.TemporaryDirectory(prefix="tt-rqm-h2b-")
        if retained_work_dir is None
        else nullcontext(str(retained_work_dir.resolve()))
    )
    with context as temporary:
        work_dir = Path(temporary)
        if retained_work_dir is not None:
            if work_dir.exists() and any(work_dir.iterdir()):
                raise HamiltonianEvolutionCandidateError(
                    "retained candidate work directory must be new or empty"
                )
            work_dir.mkdir(parents=True, exist_ok=True)
        coefficients_path = work_dir / "hamiltonians.bin"
        dt_path = work_dir / "dt.bin"
        _write_float32(coefficients_path, coefficients)
        _write_float32(dt_path, dt_value)
        manifest = {
            "schema": PROTOCOL,
            "benchmark": "HamiltonianEvolutionBench",
            "stage": stage,
            "dtype": "float32",
            "hamiltonian_shape": list(shape),
            "hamiltonian_lane_order": ["h0", "hx", "hy", "hz"],
            "dt_shape": list(dt_value.shape),
            "hbar": float(hbar),
            "final_rotor_shape": [shape[0], 4],
            "final_rotor_lane_order": ["w", "x", "y", "z"],
            "final_phase_shape": [shape[0], 2],
            "final_phase_lane_order": ["real", "imag"],
            "logical_order": "row-major",
            "format": "raw little-endian float32 row-major",
            "inputs": {
                "hamiltonians": coefficients_path.name,
                "dt": dt_path.name,
                "hamiltonians_sha256": _sha256(coefficients_path),
                "dt_sha256": _sha256(dt_path),
            },
            "outputs": {
                "final_rotors": "final_rotors.bin",
                "final_phases": "final_phases.bin",
                "metrics": "metrics.json",
            },
        }
        manifest_path = work_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        host_elapsed, stdout, stderr = _execute(command, work_dir, manifest_path)
        metrics = _load_metrics(work_dir / "metrics.json")
        _validate_metrics(metrics, manifest, execution_label=execution_label)
        final_rotors = _read_float32(work_dir / "final_rotors.bin", (shape[0], 4))
        final_phases = _read_float32(work_dir / "final_phases.bin", (shape[0], 2))

    correctness = _validate_outputs(
        final_rotors,
        final_phases,
        coefficients,
        dt_value,
        hbar,
        atol=HARDWARE_ATOL if execution_label == "hardware" else ATOL,
        rtol=HARDWARE_RTOL if execution_label == "hardware" else RTOL,
    )
    report = {
        "schema": REPORT_SCHEMA,
        "protocol": PROTOCOL,
        "benchmark_family": "HamiltonianEvolutionBench",
        "stage": stage,
        "execution_label": execution_label,
        "dtype": "float32",
        "B": shape[0],
        "K": shape[1],
        "input_hashes": {
            "hamiltonians_sha256": hashlib.sha256(coefficients.numpy().tobytes()).hexdigest(),
            "dt_sha256": hashlib.sha256(dt_value.numpy().tobytes()).hexdigest(),
        },
        "correctness": correctness,
        "candidate_metrics": metrics,
        "host_end_to_end_s": host_elapsed,
        "stable_benchmark": False,
        "performance_eligible": False,
        "claim_level": None,
    }
    return CandidateRun(report, final_rotors, final_phases, stdout, stderr)


def deterministic_candidate_inputs(
    *, seed: int = 0, B: int = 4, K: int = 8
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a deterministic mixed H2B conformance input."""

    if B < 1 or K < 1:
        raise ValueError("candidate inputs require B >= 1 and K >= 1")
    generator = torch.Generator().manual_seed(seed)
    coefficients = 0.5 * torch.randn((B, K, 4), generator=generator, dtype=torch.float32)
    coefficients[0, 0, 1:] = 0.0
    if K > 1:
        coefficients[0, 1, 1:] = torch.tensor((1e-8, -1e-8, 1e-8))
    dt = torch.linspace(-0.1, 0.4, B * K, dtype=torch.float32).reshape(B, K)
    return coefficients, dt


def _validate_outputs(
    rotors: torch.Tensor,
    phases: torch.Tensor,
    coefficients: torch.Tensor,
    dt: torch.Tensor,
    hbar: float,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    nonfinite = int((~torch.isfinite(rotors)).sum() + (~torch.isfinite(phases)).sum())
    if nonfinite:
        raise HamiltonianEvolutionCandidateError(
            f"candidate output contains {nonfinite} nonfinite values"
        )
    reference_rotor, reference_phase = evolve_two_level_hamiltonian(
        coefficients.double(), dt.double(), hbar=hbar
    )
    rotor64, phase64 = rotors.double(), phases.double()
    rotor_error = torch.abs(rotor64 - reference_rotor)
    phase_error = torch.abs(phase64 - reference_phase)
    rotor_relative = rotor_error / torch.clamp(torch.abs(reference_rotor), min=1e-30)
    phase_relative = phase_error / torch.clamp(torch.abs(reference_phase), min=1e-30)
    rotor_fail = rotor_error > (atol + rtol * torch.abs(reference_rotor))
    phase_fail = phase_error > (atol + rtol * torch.abs(reference_phase))
    failing = int(rotor_fail.sum() + phase_fail.sum())
    if failing:
        raise HamiltonianEvolutionCandidateError(
            f"candidate output failed whole-output validation at {failing} values"
        )
    actual_matrix = u2_matrix_from_rotor_phase(rotor64, phase64)
    matrix_oracle = compose_hamiltonian_matrices(coefficients, dt, hbar=hbar)
    direct = float(torch.max(torch.abs(actual_matrix - matrix_oracle)).item())
    overlap = torch.sum(torch.conj(matrix_oracle) * actual_matrix, dim=(-2, -1))
    unit_phase = torch.where(overlap.abs() > 0, overlap / overlap.abs(), torch.ones_like(overlap))
    aligned = actual_matrix * torch.conj(unit_phase)[..., None, None]
    return {
        "passed": True,
        "failing_value_count": 0,
        "nonfinite_value_count": 0,
        "max_rotor_absolute_error": float(rotor_error.max()),
        "max_rotor_relative_error": float(rotor_relative.max()),
        "max_phase_absolute_error": float(phase_error.max()),
        "max_phase_relative_error": float(phase_relative.max()),
        "rotor_norm_drift": float(
            torch.max(torch.abs(torch.linalg.vector_norm(rotor64, dim=-1) - 1.0))
        ),
        "phase_norm_drift": float(
            torch.max(torch.abs(torch.linalg.vector_norm(phase64, dim=-1) - 1.0))
        ),
        "complex128_final_matrix_error": direct,
        "direct_final_matrix_error": direct,
        "global_phase_aware_final_matrix_error": float(
            torch.max(torch.abs(aligned - matrix_oracle)).item()
        ),
        "checksum": hashlib.sha256(
            rotors.contiguous().numpy().tobytes() + phases.contiguous().numpy().tobytes()
        ).hexdigest(),
    }


def _validate_metrics(
    metrics: dict[str, Any], manifest: dict[str, Any], *, execution_label: str
) -> None:
    expected = {
        "schema": METRICS_SCHEMA,
        "protocol": PROTOCOL,
        "benchmark": "HamiltonianEvolutionBench",
        "stage": manifest["stage"],
        "dtype": "float32",
        "execution_label": execution_label,
        "hamiltonian_shape": manifest["hamiltonian_shape"],
        "dt_shape": manifest["dt_shape"],
        "final_rotor_shape": manifest["final_rotor_shape"],
        "final_phase_shape": manifest["final_phase_shape"],
        "final_rotor_lane_order": ["w", "x", "y", "z"],
        "final_phase_lane_order": ["real", "imag"],
        "stable_benchmark": False,
        "performance_eligible": False,
        "claim_level": None,
    }
    for key, value in expected.items():
        if metrics.get(key) != value:
            raise HamiltonianEvolutionCandidateError(
                f"candidate metrics mismatch for {key}: expected {value!r}"
            )
    timings = metrics.get("timings_s")
    if not isinstance(timings, dict) or not timings:
        raise HamiltonianEvolutionCandidateError("candidate metrics require timings_s")
    if any(
        type(value) not in {int, float} or not math.isfinite(value) or value < 0.0
        for value in timings.values()
    ):
        raise HamiltonianEvolutionCandidateError("candidate timings must be finite and nonnegative")
    metadata = metrics.get("candidate_metadata")
    if not isinstance(metadata, dict):
        raise HamiltonianEvolutionCandidateError("candidate_metadata must be an object")
    if execution_label == "cpu_reference":
        if metadata.get("implementation_class") != "cpu_reference":
            raise HamiltonianEvolutionCandidateError(
                "CPU candidate must use implementation_class=cpu_reference"
            )
        if metadata.get("device") != "cpu/pytorch-reference":
            raise HamiltonianEvolutionCandidateError(
                "CPU candidate must use device=cpu/pytorch-reference"
            )
        return

    required_strings = {
        "implementation_class",
        "candidate_sha256",
        "source_commit",
        "source_bundle_sha256",
        "tt_metal_commit",
        "compiler_version",
        "runtime_version",
        "device_arch",
        "input_layout",
        "intermediate_layout",
        "output_layout",
        "intermediate_storage",
        "h2a_arithmetic_path",
        "h1_arithmetic_path",
        "composition_order",
    }
    required = required_strings | {
        "source_tree_clean",
        "device_count",
        "device_id",
        "device_create_count",
        "device_close_count",
        "program_count",
        "h2a_core_count",
        "h1_core_count",
        "device_resident_intermediate",
        "intermediate_d2h_count",
        "intermediate_h2d_count",
        "host_round_trip_count",
        "automatic_normalization",
    }
    if not required.issubset(metadata):
        raise HamiltonianEvolutionCandidateError("hardware candidate metadata is incomplete")
    if any(not isinstance(metadata[key], str) or not metadata[key] for key in required_strings):
        raise HamiltonianEvolutionCandidateError("hardware metadata strings must be nonempty")
    for key in ("candidate_sha256", "source_bundle_sha256"):
        value = metadata[key]
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise HamiltonianEvolutionCandidateError(f"hardware metadata {key} must be SHA-256")
    if len(metadata["source_commit"]) != 40:
        raise HamiltonianEvolutionCandidateError("source_commit must be a full Git commit")
    if metadata["tt_metal_commit"] != TT_METAL_COMMIT:
        raise HamiltonianEvolutionCandidateError("hardware candidate TT-Metal commit mismatch")
    expected_integers = {
        "device_count": 1,
        "device_id": 0,
        "device_create_count": 1,
        "device_close_count": 1,
        "program_count": 2,
        "h2a_core_count": 1,
        "intermediate_d2h_count": 0,
        "intermediate_h2d_count": 0,
        "host_round_trip_count": 0,
    }
    for key, value in expected_integers.items():
        if type(metadata[key]) is not int or metadata[key] != value:  # noqa: E721
            raise HamiltonianEvolutionCandidateError(f"hardware metadata {key} must equal {value}")
    if type(metadata["h1_core_count"]) is not int or metadata["h1_core_count"] < 1:  # noqa: E721
        raise HamiltonianEvolutionCandidateError("h1_core_count must be a positive integer")
    expected_values = {
        "intermediate_storage": "device_dram",
        "device_resident_intermediate": True,
        "composition_order": "K-1 ... 0",
        "automatic_normalization": False,
    }
    for key, value in expected_values.items():
        if metadata.get(key) != value:
            raise HamiltonianEvolutionCandidateError(
                f"hardware metadata {key} must equal {value!r}"
            )
    if "wormhole" not in metadata["device_arch"].lower():
        raise HamiltonianEvolutionCandidateError("hardware candidate must identify Wormhole")
    if (
        "cpu" in metadata["h2a_arithmetic_path"].lower()
        or "cpu" in metadata["h1_arithmetic_path"].lower()
    ):
        raise HamiltonianEvolutionCandidateError("hardware arithmetic paths must not use CPU")


def _execute(command: str, work_dir: Path, manifest: Path) -> tuple[float, str, str]:
    env = os.environ.copy()
    env["TT_RQM_H2B_DIR"] = str(work_dir)
    env["TT_RQM_H2B_MANIFEST"] = str(manifest)
    started = time.perf_counter()
    completed = subprocess.run(shlex.split(command), capture_output=True, text=True, env=env)
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise HamiltonianEvolutionCandidateError(
            f"candidate command failed\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return elapsed, completed.stdout, completed.stderr


def _load_metrics(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise HamiltonianEvolutionCandidateError("candidate did not write metrics.json")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HamiltonianEvolutionCandidateError("candidate metrics JSON is malformed") from exc
    if not isinstance(payload, dict):
        raise HamiltonianEvolutionCandidateError("candidate metrics must be an object")
    return payload


def _write_float32(path: Path, value: torch.Tensor) -> None:
    payload = array("f", value.reshape(-1).tolist())
    if sys.byteorder != "little":
        payload.byteswap()
    path.write_bytes(payload.tobytes())


def _read_float32(path: Path, shape: tuple[int, ...]) -> torch.Tensor:
    if not path.is_file():
        raise HamiltonianEvolutionCandidateError(f"candidate did not write {path.name}")
    payload = path.read_bytes()
    expected_bytes = math.prod(shape) * 4
    if len(payload) != expected_bytes:
        raise HamiltonianEvolutionCandidateError(
            f"{path.name} has {len(payload)} bytes; expected {expected_bytes}"
        )
    values = array("f")
    values.frombytes(payload)
    if sys.byteorder != "little":
        values.byteswap()
    return torch.tensor(values, dtype=torch.float32).reshape(shape)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
