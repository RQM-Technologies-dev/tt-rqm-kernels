from __future__ import annotations

import copy

import pytest
import torch

from tt_rqm_kernels.qsp_benchmark import (
    FORMULATIONS,
    OPERATORS,
    collect_cpu_report,
    qualify_hardware_sessions,
    validate_hardware_report,
)
from tt_rqm_kernels.qsp_kernels import make_operator_inputs, run_operator


@pytest.mark.parametrize("operator", OPERATORS)
def test_every_formulation_matches_whole_output(operator: str) -> None:
    tensors = make_operator_inputs(operator, 17, seed=20260808, dtype=torch.float64)
    golden = run_operator(operator, "unrestricted", tensors)
    for formulation in FORMULATIONS:
        actual = run_operator(operator, formulation, tensors)
        torch.testing.assert_close(actual, golden, rtol=1e-11, atol=1e-11)


def test_cpu_report_is_matched_and_never_grants_hardware_claim() -> None:
    report = collect_cpu_report(sizes=(8,), warmups=0, repetitions=3)
    assert report["execution_label"] == "cpu"
    assert report["hardware_gate_status"] == "pending_hardware"
    assert report["claim_flags"] == {
        "application_acceleration_claim_allowed": False,
        "native_quaternion_execution_claim_allowed": False,
        "structured_kernel_advantage_allowed": False,
    }
    assert len(report["results"]) == len(OPERATORS) * len(FORMULATIONS)
    assert all(row["correctness"]["passed"] for row in report["results"])


def _hardware_report(session_id: str) -> dict[str, object]:
    return {
        "benchmark_stage": "performance",
        "correctness_passed": True,
        "execution_label": "hardware",
        "native_gate_passed": True,
        "operators": list(OPERATORS),
        "provenance": {
            "build_id": "build-1",
            "candidate_sha256": "a" * 64,
            "chip_type": "Wormhole n300",
            "compiler_version": "compiler-1",
            "runtime_version": "runtime-1",
            "timer_scope": "device synchronized",
            "tt_metal_commit": "b" * 40,
        },
        "results": [
            {
                "correctness": {"passed": True},
                "operator": operator,
                "samples_s": [0.001 + index * 1e-6 for index in range(10)],
            }
            for operator in OPERATORS
        ],
        "schema": "tt-rqm-qsp-kernel-benchmark.v1",
        "session_id": session_id,
        "stable_benchmark": False,
        "structured_gate_passed": True,
    }


def test_hardware_contract_requires_three_distinct_real_sessions() -> None:
    report = _hardware_report("session-1")
    validate_hardware_report(report, expected_stage="performance")  # type: ignore[arg-type]
    sessions = [
        _hardware_report("session-1"),
        _hardware_report("session-2"),
        _hardware_report("session-3"),
    ]
    qualification = qualify_hardware_sessions(sessions)  # type: ignore[arg-type]
    assert qualification["stable_benchmark"] is True
    assert qualification["native_gate_passed"] is True
    duplicated = copy.deepcopy(sessions)
    duplicated[2]["session_id"] = "session-2"
    with pytest.raises(ValueError, match="distinct"):
        qualify_hardware_sessions(duplicated)  # type: ignore[arg-type]
