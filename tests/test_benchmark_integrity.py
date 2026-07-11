from __future__ import annotations

import sys

import pytest
import torch

from tt_rqm_kernels.benchmark_integrity import (
    IntegrityError,
    independent_qmul_golden,
    timing_summary,
    validate_execution_policy,
    validate_external_metrics,
    validate_qmul_output,
    validate_report,
)
from tt_rqm_kernels.structuredbench import run_suite


FAST_EXTERNAL_QMUL = f"{sys.executable} tests/fixtures/qmul_external_fast.py"


def test_golden_uses_exact_float32_inputs_promoted_to_float64() -> None:
    a = torch.tensor([[1.00000006, 0.25, -0.5, 0.75]], dtype=torch.float64)
    b = torch.tensor([[0.5, -0.25, 0.125, 0.75]], dtype=torch.float64)
    golden = independent_qmul_golden(a, b)
    rounded = a.to(torch.float32).to(torch.float64)

    assert golden.dtype == torch.float64
    assert torch.equal(golden, independent_qmul_golden(rounded, b.to(torch.float32)))


def test_whole_output_rejects_corruption_after_scalar_diagnostic_window() -> None:
    a = torch.zeros((16, 4), dtype=torch.float32)
    b = torch.zeros((16, 4), dtype=torch.float32)
    output = independent_qmul_golden(a, b).to(torch.float32)
    output[8, 0] = 1.0

    with pytest.raises(IntegrityError, match="whole-output validation failed"):
        validate_qmul_output(output, a, b)


def test_whole_output_rejects_nonfinite_values() -> None:
    a = torch.zeros((2, 4), dtype=torch.float32)
    output = torch.zeros((2, 4), dtype=torch.float32)
    output[-1, -1] = torch.nan

    with pytest.raises(IntegrityError, match="non-finite"):
        validate_qmul_output(output, a, a)


@pytest.mark.parametrize(
    ("env_name", "message"),
    [
        ("TT_RQM_TEST_CORRUPT_AFTER_EIGHT", "whole-output validation failed"),
        ("TT_RQM_TEST_NAN_OUTPUT", "non-finite"),
        ("TT_RQM_TEST_MISMATCH_METRICS", "items mismatch"),
        ("TT_RQM_TEST_FABRICATED_TIMING", "exceeds host end-to-end"),
    ],
)
def test_external_candidate_adversarial_outputs_are_rejected(
    monkeypatch: pytest.MonkeyPatch, env_name: str, message: str
) -> None:
    monkeypatch.setenv(env_name, "1")
    with pytest.raises(IntegrityError, match=message):
        run_suite(
            "qmul",
            backend="external-qmul",
            external_command=FAST_EXTERNAL_QMUL,
            execution_label="cpu",
            items_override=16,
            iterations_override=1,
            warmup_override=0,
        )


def test_timing_summary_uses_median_and_nearest_rank_p95() -> None:
    summary = timing_summary([10.0, 1.0, 5.0, 3.0, 2.0])

    assert summary["median"] == 3.0
    assert summary["p95"] == 10.0


def test_external_repetitions_emit_raw_samples_median_and_p95() -> None:
    report = run_suite(
        "qmul",
        backend="external-qmul",
        external_command=FAST_EXTERNAL_QMUL,
        execution_label="cpu",
        items_override=16,
        iterations_override=1,
        warmup_override=0,
        repetitions=3,
    )

    timing = report["results"][0]["timing"]
    assert timing["repetitions"] == 3
    assert len(timing["device_s"]["samples"]) == 3
    assert timing["device_s"]["median"] > 0
    assert timing["device_s"]["p95"] >= timing["device_s"]["median"]


def test_stage_constraints_and_fake_hardware_labels_are_rejected() -> None:
    with pytest.raises(IntegrityError, match="real Tenstorrent"):
        validate_execution_policy(
            backend="external-qmul",
            execution_label="hardware",
            stable_benchmark=False,
            command="docker run tt-emule",
            stage="conformance",
            repetitions=1,
            items=[128],
        )
    with pytest.raises(IntegrityError, match="at least 10 repetitions"):
        validate_execution_policy(
            backend="external-qmul",
            execution_label="hardware",
            stable_benchmark=False,
            command="/opt/tt/qmul_hw",
            stage="performance",
            repetitions=9,
            items=[4096, 65536, 262144],
        )
    with pytest.raises(IntegrityError, match="stable benchmark"):
        validate_execution_policy(
            backend="tt-lang-sim",
            execution_label="simulator",
            stable_benchmark=True,
            items=[128],
        )


def test_performance_stage_rejects_scalar_correctness_baseline() -> None:
    manifest = {
        "schema": "tt-rqm-external-qmul.v1",
        "dtype": "float32",
        "items": 4096,
        "iterations": 30,
        "warmup": 5,
    }
    metrics = {
        "schema": "tt-rqm-external-qmul-metrics.v2",
        "protocol": "tt-rqm-external-qmul.v1",
        "backend": "tt-metalium-qmul-riscv-candidate",
        "device": "tenstorrent/wormhole",
        "dtype": "float32",
        "items": 4096,
        "iterations": 30,
        "warmup": 5,
        "execution_kind": "hardware",
        "implementation_class": "scalar_riscv_correctness_baseline",
        "performance_eligible": False,
        "timings_s": {"setup": 0.1, "device": 0.2},
        "provenance": {
            "chip_type": "wormhole",
            "tt_metal_commit": "abc",
            "compiler_version": "c++",
            "runtime_version": "tt-metal",
            "build_id": "build",
            "timer_scope": "enqueue plus finish",
        },
    }

    with pytest.raises(IntegrityError, match="performance_eligible"):
        validate_external_metrics(
            metrics,
            manifest,
            execution_label="hardware",
            host_end_to_end_s=1.0,
            candidate_sha256="hash",
            stage="performance",
        )


def test_hardware_metrics_require_full_provenance() -> None:
    manifest = {
        "schema": "tt-rqm-external-qmul.v1",
        "dtype": "float32",
        "items": 128,
        "iterations": 1,
        "warmup": 0,
    }
    metrics = {
        "schema": "tt-rqm-external-qmul-metrics.v2",
        "protocol": "tt-rqm-external-qmul.v1",
        "backend": "candidate",
        "device": "tenstorrent/wormhole",
        "dtype": "float32",
        "items": 128,
        "iterations": 1,
        "warmup": 0,
        "execution_kind": "hardware",
        "implementation_class": "scalar_riscv_correctness_baseline",
        "performance_eligible": False,
        "timings_s": {"setup": 0.1, "device": 0.2},
        "provenance": {"chip_type": "unknown"},
    }

    with pytest.raises(IntegrityError, match="missing provenance"):
        validate_external_metrics(
            metrics,
            manifest,
            execution_label="hardware",
            host_end_to_end_s=1.0,
            candidate_sha256="hash",
            stage="conformance",
        )


def test_tt_lang_failed_correctness_cannot_be_emitted() -> None:
    report = {
        "backend": "tt-lang-sim",
        "execution_label": "simulator",
        "stable_benchmark": False,
        "repetitions": 1,
        "case_items": [128],
        "results": [
            {
                "iterations": 1,
                "warmup": 0,
                "elapsed_s": 1.0,
                "latency_ms": 1000.0,
                "throughput": 128.0,
                "max_abs_error": 1.0,
                "rms_error": 1.0,
                "correctness": {"passed": False},
            }
        ],
    }

    with pytest.raises(IntegrityError, match="pass correctness"):
        validate_report(report)
