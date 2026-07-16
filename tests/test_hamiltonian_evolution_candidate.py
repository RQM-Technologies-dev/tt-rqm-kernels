from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

from tt_rqm_kernels.hamiltonian_evolution_candidate import (
    HamiltonianEvolutionCandidateError,
    deterministic_candidate_inputs,
    run_external_candidate,
)

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = f"{sys.executable} scripts/hamiltonian_evolution_external_reference.py"
FAULTY = f"{sys.executable} tests/fixtures/hamiltonian_evolution_candidate_faulty.py"


def test_external_cpu_reference_candidate_passes() -> None:
    coefficients, dt = deterministic_candidate_inputs(seed=2, B=3, K=8)
    run = run_external_candidate(coefficients, dt, command=REFERENCE)
    assert run.report["correctness"]["passed"] is True
    assert run.final_rotors.shape == (3, 4)
    assert run.final_phases.shape == (3, 2)
    assert run.report["stable_benchmark"] is False
    assert run.report["performance_eligible"] is False
    assert run.report["claim_level"] is None


def test_external_protocol_supports_k1_and_scalar_dt() -> None:
    coefficients = torch.zeros((2, 1, 4), dtype=torch.float32)
    run = run_external_candidate(coefficients, 0.25, command=REFERENCE)
    assert torch.equal(run.final_rotors[:, 0], torch.ones(2))


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("malformed_metrics", "malformed"),
        ("wrong_lane_order", "lane_order"),
        ("truncate_rotor", "final_rotors.bin has"),
        ("truncate_phase", "final_phases.bin has"),
        ("reorder", "whole-output validation"),
        ("nan", "nonfinite"),
        ("inf", "nonfinite"),
        ("stable", "stable_benchmark"),
        ("performance", "performance_eligible"),
        ("claim", "claim_level"),
    ],
)
def test_external_protocol_fails_closed_on_output_and_claim_faults(
    monkeypatch: pytest.MonkeyPatch, fault: str, message: str
) -> None:
    monkeypatch.setenv("TT_RQM_H2B_TEST_FAULT", fault)
    coefficients, dt = deterministic_candidate_inputs(B=2, K=8)
    with pytest.raises(HamiltonianEvolutionCandidateError, match=message):
        run_external_candidate(coefficients, dt, command=FAULTY)


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("device_metadata", "device_id"),
        ("host_round_trip", "host_round_trip_count"),
        ("intermediate_d2h", "intermediate_d2h_count"),
        ("intermediate_h2d", "intermediate_h2d_count"),
        ("tt_metal_commit", "TT-Metal"),
    ],
)
def test_hardware_metadata_fails_closed(
    monkeypatch: pytest.MonkeyPatch, fault: str, message: str
) -> None:
    monkeypatch.setenv("TT_RQM_H2B_TEST_HARDWARE", "1")
    monkeypatch.setenv("TT_RQM_H2B_TEST_FAULT", fault)
    coefficients, dt = deterministic_candidate_inputs(B=2, K=8)
    with pytest.raises(HamiltonianEvolutionCandidateError, match=message):
        run_external_candidate(coefficients, dt, command=FAULTY, execution_label="hardware")


def test_hardware_metadata_accepts_two_program_device_resident_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TT_RQM_H2B_TEST_HARDWARE", "1")
    coefficients, dt = deterministic_candidate_inputs(B=2, K=8)
    run = run_external_candidate(coefficients, dt, command=FAULTY, execution_label="hardware")
    metadata = run.report["candidate_metrics"]["candidate_metadata"]
    assert metadata["program_count"] == 2
    assert metadata["device_resident_intermediate"] is True
    assert metadata["host_round_trip_count"] == 0
