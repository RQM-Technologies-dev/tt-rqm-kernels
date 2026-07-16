from __future__ import annotations

import math
from pathlib import Path
import sys

import pytest
import torch

from tt_rqm_kernels.hamiltonian_lowering_candidate import (
    HamiltonianLoweringCandidateError,
    _validate_metrics,
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


def _hardware_metrics() -> tuple[dict[str, object], dict[str, object]]:
    manifest = {
        "stage": "conformance",
        "hamiltonian_shape": [1, 2, 4],
        "dt_shape": [],
    }
    metadata = {
        "implementation_class": "single_core_tensix_sfpu_h2a",
        "candidate_sha256": "a" * 64,
        "source_commit": "b" * 40,
        "source_tree_clean": False,
        "source_bundle_sha256": "c" * 64,
        "tt_metal_commit": "dd2849b5bc6b7a5d38a9eafbeba31ef8d530f8d4",
        "compiler_version": "c++ 11.4",
        "runtime_version": "tt-metal pinned",
        "device_count": 1,
        "device_id": 0,
        "device_arch": "wormhole_b0",
        "core_count": 1,
        "arithmetic_path": "single Tensix FP32 SFPU",
        "input_layout": "component planar",
        "output_layout": "component planar",
        "scalar_dt_expansion": True,
        "sfpu_sqrt_mode": "sqrt_tile<false>",
        "sfpu_reciprocal_mode": "recip_tile<false>",
        "sfpu_sine_mode": "sin_tile",
        "sfpu_cosine_mode": "cos_tile",
        "zero_mask_strategy": "eqz then safe denominator",
        "device_create_count": 1,
        "device_close_count": 1,
    }
    metrics = {
        "schema": "tt-rqm-external-hamiltonian-lowering-metrics.v1",
        "protocol": "tt-rqm-external-hamiltonian-lowering.v1",
        "benchmark": "HamiltonianLoweringBench",
        "stage": "conformance",
        "dtype": "float32",
        "execution_label": "hardware",
        "stable_benchmark": False,
        "performance_eligible": False,
        "hamiltonian_shape": [1, 2, 4],
        "dt_shape": [],
        "timings_s": {"execute": 0.1},
        "candidate_metadata": metadata,
    }
    return metrics, manifest


def test_hardware_metadata_contract_accepts_complete_real_device_identity() -> None:
    metrics, manifest = _hardware_metrics()
    _validate_metrics(metrics, manifest, execution_label="hardware")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (("candidate_sha256", None), "incomplete"),
        (("device_id", 1), "device_id"),
        (("core_count", 2), "core_count"),
        (("tt_metal_commit", "d" * 40), "TT-Metal"),
        (("arithmetic_path", "cpu fallback"), "CPU"),
    ],
)
def test_hardware_metadata_contract_fails_closed(
    mutation: tuple[str, object], message: str
) -> None:
    metrics, manifest = _hardware_metrics()
    key, value = mutation
    metadata = metrics["candidate_metadata"]
    assert isinstance(metadata, dict)
    if value is None:
        del metadata[key]
    else:
        metadata[key] = value
    with pytest.raises(HamiltonianLoweringCandidateError, match=message):
        _validate_metrics(metrics, manifest, execution_label="hardware")
