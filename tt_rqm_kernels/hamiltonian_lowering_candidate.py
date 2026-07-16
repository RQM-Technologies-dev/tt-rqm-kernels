"""Fail-closed external-candidate protocol for H2A coefficient lowering."""

from __future__ import annotations

from array import array
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

from tt_rqm_kernels.hamiltonian.su2_lowering import lower_two_level_hamiltonian
from tt_rqm_kernels.hamiltonian_lowering_benchmark import (
    ATOL,
    RTOL,
    analytical_lowering_oracle,
    matrix_exp_step_oracle,
    rotor_phase_matrix,
)

PROTOCOL = "tt-rqm-external-hamiltonian-lowering.v1"
METRICS_SCHEMA = "tt-rqm-external-hamiltonian-lowering-metrics.v1"
Stage = Literal["conformance", "performance"]
ExecutionLabel = Literal["cpu_reference", "hardware"]


class HamiltonianLoweringCandidateError(RuntimeError):
    """Raised when an H2A candidate violates the serialized protocol."""


@dataclass(frozen=True)
class CandidateRun:
    report: dict[str, Any]
    rotors: torch.Tensor
    phases: torch.Tensor


def run_external_candidate(
    hamiltonians: torch.Tensor,
    dt: float | torch.Tensor,
    *,
    command: str,
    stage: Stage = "conformance",
    execution_label: ExecutionLabel = "cpu_reference",
    hbar: float = 1.0,
) -> CandidateRun:
    """Serialize inputs, execute one candidate, and validate every returned value."""

    if stage not in {"conformance", "performance"}:
        raise HamiltonianLoweringCandidateError(f"unsupported stage: {stage}")
    if execution_label not in {"cpu_reference", "hardware"}:
        raise HamiltonianLoweringCandidateError(f"unsupported execution label: {execution_label}")
    if not command or not shlex.split(command):
        raise HamiltonianLoweringCandidateError("candidate command must not be empty")
    try:
        lower_two_level_hamiltonian(hamiltonians, dt, hbar=hbar)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise HamiltonianLoweringCandidateError(str(exc)) from exc
    if hamiltonians.dtype != torch.float32:
        raise HamiltonianLoweringCandidateError("H2A external inputs must use float32")
    coefficients = hamiltonians.detach().cpu().contiguous()
    dt_value = torch.as_tensor(dt, dtype=torch.float32).detach().cpu().contiguous()
    shape = tuple(int(value) for value in coefficients.shape)

    with tempfile.TemporaryDirectory(prefix="tt-rqm-h2a-") as temporary:
        work_dir = Path(temporary)
        coefficients_path = work_dir / "hamiltonians.bin"
        dt_path = work_dir / "dt.bin"
        _write_float32(coefficients_path, coefficients)
        _write_float32(dt_path, dt_value)
        manifest = {
            "schema": PROTOCOL,
            "benchmark": "HamiltonianLoweringBench",
            "stage": stage,
            "dtype": "float32",
            "hamiltonian_shape": list(shape),
            "hamiltonian_lane_order": ["h0", "hx", "hy", "hz"],
            "dt_shape": list(dt_value.shape),
            "hbar": float(hbar),
            "rotor_shape": [shape[0], shape[1], 4],
            "rotor_lane_order": ["w", "x", "y", "z"],
            "phase_shape": [shape[0], shape[1], 2],
            "phase_lane_order": ["real", "imag"],
            "format": "raw little-endian float32 row-major",
            "inputs": {
                "hamiltonians": coefficients_path.name,
                "dt": dt_path.name,
                "hamiltonians_sha256": _sha256(coefficients_path),
                "dt_sha256": _sha256(dt_path),
            },
            "outputs": {
                "rotors": "rotors.bin",
                "phases": "phases.bin",
                "metrics": "metrics.json",
            },
        }
        manifest_path = work_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        host_elapsed = _execute(command, work_dir, manifest_path)
        metrics = _load_metrics(work_dir / "metrics.json")
        _validate_metrics(metrics, manifest, execution_label=execution_label)
        rotors = _read_float32(work_dir / "rotors.bin", (shape[0], shape[1], 4))
        phases = _read_float32(work_dir / "phases.bin", (shape[0], shape[1], 2))

    reference_rotors, reference_phases = analytical_lowering_oracle(
        coefficients, dt_value, hbar=hbar
    )
    correctness = _validate_outputs(
        rotors, phases, reference_rotors, reference_phases, coefficients, dt_value, hbar
    )
    report = {
        "schema": "tt-rqm-hamiltonian-lowering-candidate-report.v1",
        "protocol": PROTOCOL,
        "benchmark_family": "HamiltonianLoweringBench",
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
    return CandidateRun(report=report, rotors=rotors, phases=phases)


def deterministic_candidate_inputs(
    *, seed: int = 0, B: int = 4, K: int = 8
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a mixed deterministic conformance input with scalar and vector terms."""

    if B < 1 or K < 2:
        raise ValueError("candidate inputs require B >= 1 and K >= 2")
    generator = torch.Generator().manual_seed(seed)
    coefficients = torch.randn((B, K, 4), generator=generator, dtype=torch.float32)
    coefficients[0, 0, 1:] = 0.0
    coefficients[0, 1, 1:] = torch.tensor((1e-8, -1e-8, 1e-8))
    dt = torch.linspace(0.0, 0.4, B * K, dtype=torch.float32).reshape(B, K)
    return coefficients, dt


def _validate_outputs(
    rotors: torch.Tensor,
    phases: torch.Tensor,
    reference_rotors: torch.Tensor,
    reference_phases: torch.Tensor,
    coefficients: torch.Tensor,
    dt: torch.Tensor,
    hbar: float,
) -> dict[str, Any]:
    nonfinite = int((~torch.isfinite(rotors)).sum().item() + (~torch.isfinite(phases)).sum().item())
    if nonfinite:
        raise HamiltonianLoweringCandidateError(
            f"candidate output contains {nonfinite} nonfinite values"
        )
    rotor64, phase64 = rotors.double(), phases.double()
    rotor_error = torch.abs(rotor64 - reference_rotors)
    phase_error = torch.abs(phase64 - reference_phases)
    rotor_fail = rotor_error > (ATOL + RTOL * torch.abs(reference_rotors))
    phase_fail = phase_error > (ATOL + RTOL * torch.abs(reference_phases))
    failing = int(rotor_fail.sum().item() + phase_fail.sum().item())
    if failing:
        raise HamiltonianLoweringCandidateError(
            f"candidate output failed whole-output validation at {failing} values"
        )
    matrix_error = float(
        torch.max(
            torch.abs(
                rotor_phase_matrix(rotor64, phase64)
                - matrix_exp_step_oracle(coefficients, dt, hbar=hbar)
            )
        ).item()
    )
    return {
        "passed": True,
        "failing_value_count": 0,
        "nonfinite_value_count": 0,
        "max_rotor_absolute_error": float(rotor_error.max().item()),
        "max_phase_absolute_error": float(phase_error.max().item()),
        "rotor_norm_drift": float(
            torch.max(torch.abs(torch.linalg.vector_norm(rotor64, dim=-1) - 1.0)).item()
        ),
        "phase_norm_drift": float(
            torch.max(torch.abs(torch.linalg.vector_norm(phase64, dim=-1) - 1.0)).item()
        ),
        "complex_matrix_reconstruction_error": matrix_error,
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
        "benchmark": "HamiltonianLoweringBench",
        "stage": manifest["stage"],
        "dtype": "float32",
        "execution_label": execution_label,
        "stable_benchmark": False,
        "performance_eligible": False,
    }
    for key, value in expected.items():
        if metrics.get(key) != value:
            raise HamiltonianLoweringCandidateError(
                f"candidate metrics mismatch for {key}: expected {value!r}"
            )
    if metrics.get("hamiltonian_shape") != manifest["hamiltonian_shape"]:
        raise HamiltonianLoweringCandidateError("candidate metrics Hamiltonian shape mismatch")
    if metrics.get("dt_shape") != manifest["dt_shape"]:
        raise HamiltonianLoweringCandidateError("candidate metrics dt shape mismatch")
    timings = metrics.get("timings_s")
    if not isinstance(timings, dict) or not timings:
        raise HamiltonianLoweringCandidateError("candidate metrics require timings_s")
    if any(
        not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0.0
        for value in timings.values()
    ):
        raise HamiltonianLoweringCandidateError("candidate timings must be finite and nonnegative")
    metadata = metrics.get("candidate_metadata")
    if not isinstance(metadata, dict):
        raise HamiltonianLoweringCandidateError("candidate_metadata must be an object")
    if execution_label == "cpu_reference":
        if metadata.get("implementation_class") != "cpu_reference":
            raise HamiltonianLoweringCandidateError(
                "CPU candidate must use implementation_class=cpu_reference"
            )
        if metadata.get("device") != "cpu/pytorch-reference":
            raise HamiltonianLoweringCandidateError(
                "CPU candidate must use device=cpu/pytorch-reference"
            )
    else:
        required = {
            "candidate_sha256",
            "source_commit",
            "tt_metal_commit",
            "compiler_version",
            "runtime_version",
            "device_id",
        }
        if not required.issubset(metadata):
            raise HamiltonianLoweringCandidateError("hardware candidate metadata is incomplete")


def _execute(command_text: str, work_dir: Path, manifest_path: Path) -> float:
    env = os.environ.copy()
    env["TT_RQM_H2A_DIR"] = str(work_dir)
    env["TT_RQM_H2A_MANIFEST"] = str(manifest_path)
    started = time.perf_counter()
    completed = subprocess.run(shlex.split(command_text), capture_output=True, text=True, env=env)
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise HamiltonianLoweringCandidateError(
            f"candidate command failed\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return elapsed


def _load_metrics(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise HamiltonianLoweringCandidateError("candidate did not write metrics.json")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HamiltonianLoweringCandidateError("candidate metrics JSON is malformed") from exc
    if not isinstance(payload, dict):
        raise HamiltonianLoweringCandidateError("candidate metrics must be an object")
    return payload


def _write_float32(path: Path, value: torch.Tensor) -> None:
    payload = array("f", value.reshape(-1).tolist())
    if sys.byteorder != "little":
        payload.byteswap()
    path.write_bytes(payload.tobytes())


def _read_float32(path: Path, shape: tuple[int, ...]) -> torch.Tensor:
    if not path.is_file():
        raise HamiltonianLoweringCandidateError(f"candidate did not write {path.name}")
    values = array("f")
    values.frombytes(path.read_bytes())
    if sys.byteorder != "little":
        values.byteswap()
    expected = math.prod(shape)
    if len(values) != expected:
        raise HamiltonianLoweringCandidateError(
            f"{path.name} has {len(values)} values; expected {expected}"
        )
    return torch.tensor(values, dtype=torch.float32).reshape(shape)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
