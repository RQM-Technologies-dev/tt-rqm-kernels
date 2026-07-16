from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from tt_rqm_kernels.hamiltonian_lowering_benchmark import (
    CASE_IDS,
    analytical_lowering_oracle,
    matrix_exp_step_oracle,
    reference_cases,
    rotor_phase_matrix,
    run_reference_benchmark,
)
from tt_rqm_kernels.hamiltonian_lowering_preregistration import (
    HamiltonianLoweringPreregistrationError,
    load_preregistration,
    validate_preregistration,
)

ROOT = Path(__file__).resolve().parents[1]


def test_reference_benchmark_covers_frozen_semantic_categories() -> None:
    report = run_reference_benchmark(seed=7, iterations=1)
    assert report["schema"] == "tt-rqm-hamiltonian-lowering-report.v1"
    assert report["benchmark_family"] == "HamiltonianLoweringBench"
    assert report["execution_label"] == "cpu_reference"
    assert report["stable_benchmark"] is False
    assert report["performance_eligible"] is False
    assert tuple(result["case_id"] for result in report["results"]) == CASE_IDS
    for result in report["results"]:
        assert result["nonfinite_value_count"] == 0
        assert len(result["checksum"]) == 64
        assert result["latency_ms"] >= 0.0
        assert result["throughput_coefficients_per_s"] > 0.0
        assert result["execution_label"] == "cpu_reference"


def test_reference_inputs_and_hashes_are_deterministic() -> None:
    first = reference_cases(11)
    second = reference_cases(11)
    for left, right in zip(first, second, strict=True):
        assert left["id"] == right["id"]
        assert torch.equal(left["hamiltonians"], right["hamiltonians"])
        assert torch.equal(torch.as_tensor(left["dt"]), torch.as_tensor(right["dt"]))
    report_a = run_reference_benchmark(seed=11, iterations=1)
    report_b = run_reference_benchmark(seed=11, iterations=1)
    hashes_a = [result["input_hashes"] for result in report_a["results"]]
    hashes_b = [result["input_hashes"] for result in report_b["results"]]
    assert hashes_a == hashes_b


def test_independent_analytical_and_matrix_exp_oracles_agree() -> None:
    coefficients = torch.tensor(
        [[[0.3, 0.0, 0.0, 0.0], [0.1, 1.0, -2.0, 0.5]]], dtype=torch.float32
    )
    dt = torch.tensor([[0.2, 0.05]], dtype=torch.float32)
    rotor, phase = analytical_lowering_oracle(coefficients, dt)
    reconstructed = rotor_phase_matrix(rotor, phase)
    expected = matrix_exp_step_oracle(coefficients, dt)
    assert torch.allclose(reconstructed, expected, atol=1e-12, rtol=1e-12)
    assert torch.equal(rotor[0, 0], torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64))


@pytest.mark.parametrize(
    ("coefficients", "dt", "hbar"),
    [
        (torch.zeros((2, 4)), 0.1, 1.0),
        (torch.zeros((1, 2, 4), dtype=torch.int64), 0.1, 1.0),
        (torch.full((1, 2, 4), math.nan), 0.1, 1.0),
        (torch.full((1, 2, 4), math.inf), 0.1, 1.0),
        (torch.zeros((1, 2, 4)), torch.ones((3, 2)), 1.0),
        (torch.zeros((1, 2, 4)), 0.1, 0.0),
        (torch.zeros((1, 2, 4)), 0.1, math.inf),
    ],
)
def test_production_contract_rejects_invalid_h2a_inputs(
    coefficients: torch.Tensor, dt: float | torch.Tensor, hbar: float
) -> None:
    from tt_rqm_kernels.hamiltonian.su2_lowering import lower_two_level_hamiltonian

    with pytest.raises((TypeError, ValueError)):
        lower_two_level_hamiltonian(coefficients, dt, hbar=hbar)


def test_h2a_preregistration_is_pre_hardware_and_fail_closed() -> None:
    manifest = load_preregistration(
        ROOT / "benchmarks/manifests/hamiltonian-lowering-h2a-preregistration.json"
    )
    assert manifest["status"] == "pre_hardware"
    assert manifest["target_claim_level"] == 0
    assert manifest["claims"]["stable_benchmark"] is False
    changed = json.loads(json.dumps(manifest))
    changed["claims"]["current_level"] = 0
    with pytest.raises(HamiltonianLoweringPreregistrationError):
        validate_preregistration(changed)


def test_reference_cli_writes_machine_readable_report(tmp_path: Path) -> None:
    output = tmp_path / "reference.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/run_hamiltonian_lowering_reference.py",
            "--iterations",
            "1",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(output.read_text())["execution_label"] == "cpu_reference"
