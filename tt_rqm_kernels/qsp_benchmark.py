"""Fail-closed CPU and hardware evidence contracts for QKERNEL-1."""

from __future__ import annotations

import json
import math
import os
import platform
import statistics
import subprocess
import time
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch

from tt_rqm_kernels.qsp_kernels import (
    Formulation,
    coefficient_counts,
    make_operator_inputs,
    run_operator,
)

OPERATORS = (
    "qmul",
    "qdot",
    "su2-frame-transport",
    "selective-q-plus-qj",
    "fused-frame-filter",
    "ordered-rotor-composition",
)
FORMULATIONS: tuple[Formulation, ...] = (
    "unrestricted",
    "materialized_structured",
    "direct_structured_real",
    "complex_pair",
    "quaternion",
)
SCHEMA = "tt-rqm-qsp-kernel-benchmark.v1"


def collect_cpu_report(
    *,
    sizes: tuple[int, ...] = (256, 4096),
    warmups: int = 3,
    repetitions: int = 10,
    seed: int = 20260808,
    threads: int = 1,
) -> dict[str, Any]:
    """Collect matched vectorized PyTorch CPU correctness and timing samples."""

    if not sizes or min(sizes) <= 0 or warmups < 0 or repetitions < 3 or threads <= 0:
        raise ValueError("invalid CPU benchmark shape or timing configuration")
    torch.set_num_threads(threads)
    results: list[dict[str, Any]] = []
    for operator_index, operator in enumerate(OPERATORS):
        for size in sizes:
            tensors = make_operator_inputs(operator, size, seed=seed + operator_index * 1000 + size)
            golden = run_operator(operator, "unrestricted", tensors)
            for formulation in FORMULATIONS:
                actual = run_operator(operator, formulation, tensors)
                difference = torch.abs(actual.to(torch.float64) - golden.to(torch.float64))
                max_abs = float(torch.max(difference).item()) if difference.numel() else 0.0
                if not torch.isfinite(actual).all() or max_abs > 2e-4:
                    raise ValueError(
                        f"whole-output validation failed for {operator}/{formulation}: {max_abs}"
                    )
                for _ in range(warmups):
                    run_operator(operator, formulation, tensors)
                samples: list[float] = []
                for _ in range(repetitions):
                    started = time.perf_counter_ns()
                    timed = run_operator(operator, formulation, tensors)
                    # Materialize a scalar so all formulations share a completed host boundary.
                    float(timed.reshape(-1)[0].real.item())
                    samples.append((time.perf_counter_ns() - started) / 1e9)
                ordered = sorted(samples)
                p95 = ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]
                results.append(
                    {
                        "coefficient_count": coefficient_counts(operator)[formulation],
                        "correctness": {
                            "max_abs_error": max_abs,
                            "passed": True,
                            "validated_values": actual.numel(),
                        },
                        "formulation": formulation,
                        "median_s": statistics.median(samples),
                        "operator": operator,
                        "p95_s": p95,
                        "samples_s": samples,
                        "size": size,
                        "throughput_items_per_s": size / statistics.median(samples),
                    }
                )
    ratios = _cpu_ratios(results)
    return {
        "claim_flags": {
            "application_acceleration_claim_allowed": False,
            "native_quaternion_execution_claim_allowed": False,
            "structured_kernel_advantage_allowed": False,
        },
        "cpu_matched_control_ratio": ratios,
        "environment": _environment(threads),
        "execution_label": "cpu",
        "hardware_gate_status": "pending_hardware",
        "results": results,
        "schema": SCHEMA,
        "seed": seed,
        "source_provenance": _source_provenance(),
        "stable_benchmark": False,
        "timing_scope": "steady-state host wall clock; completed output scalar materialization",
        "warmups": warmups,
    }


def validate_hardware_report(report: dict[str, Any], *, expected_stage: str) -> None:
    """Validate an externally produced whole-output N300 report."""

    required = {
        "schema": SCHEMA,
        "execution_label": "hardware",
        "benchmark_stage": expected_stage,
        "stable_benchmark": False,
    }
    for key, value in required.items():
        if report.get(key) != value:
            raise ValueError(f"hardware report {key} mismatch")
    if sorted(report.get("operators", [])) != sorted(OPERATORS):
        raise ValueError("hardware report must cover all registered operators")
    provenance = report.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("hardware report requires provenance")
    for key in (
        "chip_type",
        "tt_metal_commit",
        "compiler_version",
        "runtime_version",
        "build_id",
        "timer_scope",
        "candidate_sha256",
    ):
        if not isinstance(provenance.get(key), str) or not provenance[key].strip():
            raise ValueError(f"hardware provenance missing {key}")
    results = report.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError("hardware report requires raw results")
    for row in results:
        if not isinstance(row, dict) or row.get("correctness", {}).get("passed") is not True:
            raise ValueError("every hardware result must pass whole-output correctness")
        samples = row.get("samples_s")
        if not isinstance(samples, list) or len(samples) < (
            10 if expected_stage == "performance" else 1
        ):
            raise ValueError("hardware result has insufficient raw samples")
        if not all(math.isfinite(float(value)) and float(value) > 0.0 for value in samples):
            raise ValueError("hardware timing samples must be finite and positive")


def qualify_hardware_sessions(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Require three independent sessions before any stable performance flag."""

    if len(reports) != 3:
        raise ValueError("exactly three qualified cold-start sessions are required")
    session_ids: set[str] = set()
    hashes: list[str] = []
    for report in reports:
        validate_hardware_report(report, expected_stage="performance")
        session_id = str(report.get("session_id", ""))
        if not session_id or session_id in session_ids:
            raise ValueError("hardware sessions require distinct nonempty session_id values")
        session_ids.add(session_id)
        hashes.append(sha256(canonical_json_bytes(report)).hexdigest())
    # External collectors must include already computed matched gates. Qualification
    # combines them without silently repurposing historical qmul evidence.
    structured = all(report.get("structured_gate_passed") is True for report in reports)
    native = all(report.get("native_gate_passed") is True for report in reports)
    correctness = all(report.get("correctness_passed") is True for report in reports)
    return {
        "correctness_passed": correctness,
        "execution_label": "hardware",
        "gate_passed": bool(structured and correctness),
        "native_gate_passed": bool(native and correctness),
        "qualified_sessions": 3,
        "report_sha256": hashes,
        "schema": "tt-rqm-qsp-kernel-qualification.v1",
        "stable_benchmark": True,
    }


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def write_json_empty(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def _cpu_ratios(rows: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[tuple[str, int], dict[str, float]] = {}
    for row in rows:
        grouped.setdefault((row["operator"], row["size"]), {})[row["formulation"]] = float(
            row["throughput_items_per_s"]
        )
    ratios = [
        values["quaternion"] / values["direct_structured_real"] for values in grouped.values()
    ]
    return {
        "geometric_mean": float(math.exp(sum(math.log(value) for value in ratios) / len(ratios))),
        "minimum": min(ratios),
    }


def _environment(threads: int) -> dict[str, object]:
    return {
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "compiler": platform.python_compiler(),
        "cpu_model": platform.processor() or platform.machine(),
        "frequency_governor": "not_portably_available",
        "machine": platform.machine(),
        "operating_system": platform.platform(),
        "torch": torch.__version__,
        "torch_threads": threads,
        "vectorized_harness": True,
    }


def _source_provenance() -> dict[str, str]:
    repository = Path(__file__).resolve().parents[1]
    candidate = repository / "tt_rqm_kernels" / "qsp_kernels.py"
    completed = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("QSP kernel source commit is not a full lowercase Git object ID")
    return {
        "candidate_sha256": sha256(candidate.read_bytes()).hexdigest(),
        "repository_commit": commit,
    }
