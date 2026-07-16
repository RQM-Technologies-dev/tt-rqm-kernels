from __future__ import annotations

import math
from pathlib import Path
import sys

import pytest
import torch

from tt_rqm_kernels.hamiltonian_lowering_candidate import (
    HamiltonianLoweringCandidateError,
    deterministic_candidate_inputs,
    run_external_candidate,
)

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = f"{sys.executable} scripts/hamiltonian_lowering_external_reference.py"
FAULTY = f"{sys.executable} tests/fixtures/hamiltonian_lowering_candidate_faulty.py"


def test_external_cpu_reference_candidate_passes_whole_output_validation() -> None:
    coefficients, dt = deterministic_candidate_inputs(seed=3, B=3, K=5)
    run = run_external_candidate(coefficients, dt, command=REFERENCE)
    assert run.report["correctness"]["passed"] is True
    assert run.report["stage"] == "conformance"
    assert run.report["execution_label"] == "cpu_reference"
    assert run.report["stable_benchmark"] is False
    assert run.report["performance_eligible"] is False
    assert run.rotors.shape == (3, 5, 4)
    assert run.phases.shape == (3, 5, 2)


def test_external_protocol_supports_scalar_dt() -> None:
    coefficients = torch.zeros((2, 3, 4), dtype=torch.float32)
    coefficients[..., 0] = 0.5
    run = run_external_candidate(coefficients, 0.25, command=FAULTY)
    assert torch.equal(run.rotors[..., 0], torch.ones((2, 3)))
    assert torch.count_nonzero(run.rotors[..., 1:]) == 0


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("truncate", "has .* values; expected"),
        ("nonfinite", "nonfinite values"),
        ("reorder", "whole-output validation"),
        ("metrics", "metrics mismatch"),
        ("missing", "did not write rotors.bin"),
    ],
)
def test_external_protocol_fails_closed_on_candidate_faults(
    monkeypatch: pytest.MonkeyPatch, fault: str, message: str
) -> None:
    monkeypatch.setenv("TT_RQM_H2A_TEST_FAULT", fault)
    coefficients, dt = deterministic_candidate_inputs(B=2, K=4)
    with pytest.raises(HamiltonianLoweringCandidateError, match=message):
        run_external_candidate(coefficients, dt, command=FAULTY)


@pytest.mark.parametrize(
    ("coefficients", "dt", "hbar", "message"),
    [
        (torch.zeros((2, 4)), 0.1, 1.0, "shape"),
        (torch.zeros((1, 2, 4), dtype=torch.int64), 0.1, 1.0, "floating-point"),
        (torch.full((1, 2, 4), math.nan), 0.1, 1.0, "finite"),
        (torch.full((1, 2, 4), math.inf), 0.1, 1.0, "finite"),
        (torch.zeros((1, 2, 4)), torch.ones((3, 2)), 1.0, "broadcastable"),
        (torch.zeros((1, 2, 4)), 0.1, -1.0, "finite positive"),
    ],
)
def test_external_protocol_rejects_invalid_contracts(
    coefficients: torch.Tensor,
    dt: float | torch.Tensor,
    hbar: float,
    message: str,
) -> None:
    with pytest.raises(HamiltonianLoweringCandidateError, match=message):
        run_external_candidate(coefficients, dt, hbar=hbar, command=REFERENCE)


def test_cpu_reference_cannot_be_mislabeled_as_hardware() -> None:
    coefficients, dt = deterministic_candidate_inputs(B=2, K=4)
    with pytest.raises(HamiltonianLoweringCandidateError, match="execution_label"):
        run_external_candidate(coefficients, dt, command=FAULTY, execution_label="hardware")
