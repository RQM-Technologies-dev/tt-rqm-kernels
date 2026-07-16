#!/usr/bin/env python3
"""Retain raw H2A development outputs even when a correctness gate fails."""

from __future__ import annotations

import argparse
from array import array
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tt_rqm_kernels.hamiltonian_lowering_benchmark import (
    analytical_lowering_oracle,
    matrix_exp_step_oracle,
    reference_cases,
    rotor_phase_matrix,
)

PACKAGE = Path(__file__).resolve().parent
DEFAULT_BINARY = PACKAGE / "build" / "tt_rqm_metalium_hamiltonian_lowering_compensated_candidate"
RUNNER = PACKAGE / "run_candidate.py"
ATOL = 1e-4
RTOL = 1e-4


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_float32(path: Path, value: torch.Tensor) -> None:
    values = array("f", value.detach().cpu().contiguous().reshape(-1).tolist())
    if sys.byteorder != "little":
        values.byteswap()
    path.write_bytes(values.tobytes())


def _read_float32(path: Path, shape: tuple[int, ...]) -> torch.Tensor:
    values = array("f")
    values.frombytes(path.read_bytes())
    if sys.byteorder != "little":
        values.byteswap()
    return torch.tensor(values, dtype=torch.float32).reshape(shape)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", default="large_angles")
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    case = next((item for item in reference_cases(seed=0) if item["id"] == args.case_id), None)
    if case is None:
        raise SystemExit(f"unknown frozen case: {args.case_id}")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit("development output path must not already exist")
    output.mkdir(parents=True)

    coefficients = case["hamiltonians"].detach().cpu().contiguous()
    dt = torch.as_tensor(case["dt"], dtype=torch.float32).detach().cpu().contiguous()
    b_size, k_size, _ = coefficients.shape
    _write_float32(output / "hamiltonians.bin", coefficients)
    _write_float32(output / "dt.bin", dt)
    manifest = {
        "schema": "tt-rqm-external-hamiltonian-lowering.v1",
        "benchmark": "HamiltonianLoweringBench",
        "stage": "conformance",
        "dtype": "float32",
        "hamiltonian_shape": list(coefficients.shape),
        "hamiltonian_lane_order": ["h0", "hx", "hy", "hz"],
        "dt_shape": list(dt.shape),
        "hbar": 1.0,
        "rotor_shape": [b_size, k_size, 4],
        "rotor_lane_order": ["w", "x", "y", "z"],
        "phase_shape": [b_size, k_size, 2],
        "phase_lane_order": ["real", "imag"],
        "format": "raw little-endian float32 row-major",
        "inputs": {
            "hamiltonians": "hamiltonians.bin",
            "dt": "dt.bin",
            "hamiltonians_sha256": _sha256(output / "hamiltonians.bin"),
            "dt_sha256": _sha256(output / "dt.bin"),
        },
        "outputs": {"rotors": "rotors.bin", "phases": "phases.bin", "metrics": "metrics.json"},
    }
    _write_json(output / "manifest.json", manifest)
    env = os.environ.copy()
    env.update(
        {
            "TT_RQM_H2A_DIR": str(output),
            "TT_RQM_H2A_MANIFEST": str(output / "manifest.json"),
            "TT_RQM_H2A_BINARY": str(args.binary.expanduser().resolve()),
        }
    )
    completed = subprocess.run(
        [sys.executable, str(RUNNER)], capture_output=True, text=True, env=env, check=False
    )
    (output / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (output / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        return completed.returncode

    rotors = _read_float32(output / "rotors.bin", (b_size, k_size, 4))
    phases = _read_float32(output / "phases.bin", (b_size, k_size, 2))
    reference_rotors, reference_phases = analytical_lowering_oracle(coefficients, dt)
    rotor_error = torch.abs(rotors.double() - reference_rotors)
    phase_error = torch.abs(phases.double() - reference_phases)
    rotor_relative = rotor_error / torch.clamp(torch.abs(reference_rotors), min=torch.finfo(torch.float64).tiny)
    phase_relative = phase_error / torch.clamp(torch.abs(reference_phases), min=torch.finfo(torch.float64).tiny)
    failures = (rotor_error > ATOL + RTOL * torch.abs(reference_rotors)).sum()
    failures += (phase_error > ATOL + RTOL * torch.abs(reference_phases)).sum()
    nonfinite = (~torch.isfinite(rotors)).sum() + (~torch.isfinite(phases)).sum()
    report = {
        "schema": "tt-rqm-h2a-development-probe.v1",
        "case_id": args.case_id,
        "designated": False,
        "qualification_eligible": False,
        "performance_eligible": False,
        "stable_benchmark": False,
        "claim_level": None,
        "candidate_sha256": _sha256(args.binary.expanduser().resolve()),
        "frozen_tolerances": {"rotor_atol": ATOL, "rotor_rtol": RTOL, "phase_atol": ATOL, "phase_rtol": RTOL},
        "correctness": {
            "passed": int(failures) == 0 and int(nonfinite) == 0,
            "failing_value_count": int(failures),
            "nonfinite_value_count": int(nonfinite),
            "max_rotor_absolute_error": float(rotor_error.max()),
            "max_rotor_relative_error": float(rotor_relative.max()),
            "max_phase_absolute_error": float(phase_error.max()),
            "max_phase_relative_error": float(phase_relative.max()),
            "rotor_norm_drift": float(torch.max(torch.abs(torch.linalg.vector_norm(rotors.double(), dim=-1) - 1.0))),
            "phase_norm_drift": float(torch.max(torch.abs(torch.linalg.vector_norm(phases.double(), dim=-1) - 1.0))),
            "complex_matrix_reconstruction_error": float(
                torch.max(
                    torch.abs(
                        rotor_phase_matrix(rotors.double(), phases.double())
                        - matrix_exp_step_oracle(coefficients, dt)
                    )
                )
            ),
        },
        "rotors": rotors.tolist(),
        "phases": phases.tolist(),
        "reference_rotors": reference_rotors.tolist(),
        "reference_phases": reference_phases.tolist(),
    }
    _write_json(output / "development-report.json", report)
    print(json.dumps(report["correctness"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
