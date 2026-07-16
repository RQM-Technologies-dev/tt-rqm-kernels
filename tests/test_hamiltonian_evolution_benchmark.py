from __future__ import annotations

import torch

from tt_rqm_kernels.hamiltonian_evolution_benchmark import (
    CASE_IDS,
    reference_cases,
    run_reference_benchmark,
)


def test_h2b_reference_benchmark_covers_required_semantics_and_lengths() -> None:
    report = run_reference_benchmark(seed=4, iterations=1)
    assert report["schema"] == "tt-rqm-hamiltonian-evolution-report.v1"
    assert report["benchmark_family"] == "HamiltonianEvolutionBench"
    assert tuple(result["case_id"] for result in report["results"]) == CASE_IDS
    assert set((1, 2, 8, 32, 128, 512)).issubset(report["chain_lengths"])
    assert report["noncommuting_reversal_matrix_difference"] > 1e-6
    assert report["stable_benchmark"] is False
    assert report["performance_eligible"] is False
    assert report["claim_level"] is None
    for result in report["results"]:
        assert result["failing_value_count"] >= 0
        assert result["nonfinite_value_count"] == 0
        assert len(result["checksum"]) == 64
        assert result["complex128_final_matrix_error"] >= 0


def test_h2b_semantic_inputs_are_deterministic() -> None:
    first, second = reference_cases(9), reference_cases(9)
    for left, right in zip(first, second, strict=True):
        assert left["id"] == right["id"]
        assert torch.equal(left["hamiltonians"], right["hamiltonians"])
        assert torch.equal(torch.as_tensor(left["dt"]), torch.as_tensor(right["dt"]))
