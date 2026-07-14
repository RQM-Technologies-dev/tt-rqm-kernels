"""Persistent-device Stage B qmul protocol, execution, and integrity gates."""

from __future__ import annotations

from array import array
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import tempfile
import time
from typing import Mapping

import torch

from tt_rqm_kernels.benchmark_integrity import (
    HARDWARE_PROVENANCE_FIELDS,
    IntegrityError,
    command_sha256,
    repository_commit,
    timing_summary,
    validate_qmul_output,
    validate_report,
)

PERSISTENT_PROTOCOL = "tt-rqm-external-qmul-persistent.v1"
PERSISTENT_METRICS_SCHEMA = "tt-rqm-external-qmul-persistent-metrics.v1"
IMPLEMENTATION_CLASS = "multicore_tensix_sfpu_qmul_persistent"
DEVICE = "tenstorrent/wormhole-device-0"
PERFORMANCE_ITEMS = (4096, 65536, 262144)
PERFORMANCE_ITERATIONS = 30
PERFORMANCE_WARMUP = 5
PERFORMANCE_SAMPLES = 10


def run_persistent_qmul(
    *,
    command: str,
    benchmark_stage: str,
    methodology_note: str,
    seed: int = 0,
) -> dict[str, object]:
    """Execute all cases in one candidate process and validate every output."""

    if benchmark_stage == "conformance":
        case_specs = [(128, 1, 0, 1)]
    elif benchmark_stage == "performance":
        case_specs = [
            (items, PERFORMANCE_ITERATIONS, PERFORMANCE_WARMUP, PERFORMANCE_SAMPLES)
            for items in PERFORMANCE_ITEMS
        ]
    else:
        raise IntegrityError("persistent qmul stage must be conformance or performance")
    if not methodology_note.strip():
        raise IntegrityError("persistent hardware reports require a methodology note")

    candidate_hash = command_sha256(command, Path.cwd())
    source_commit = repository_commit(Path.cwd())
    prepared: list[tuple[dict[str, object], torch.Tensor, torch.Tensor]] = []

    with tempfile.TemporaryDirectory(prefix="tt-rqm-persistent-qmul-") as temp:
        workdir = Path(temp)
        manifest_cases: list[dict[str, object]] = []
        for items, iterations, warmup, samples in case_specs:
            generator = torch.Generator().manual_seed(seed)
            a = torch.randn((items, 4), generator=generator, dtype=torch.float32)
            b = torch.randn((items, 4), generator=generator, dtype=torch.float32)
            a_name = f"a_{items}.bin"
            b_name = f"b_{items}.bin"
            out_name = f"out_{items}.bin"
            _write_float32(workdir / a_name, a)
            _write_float32(workdir / b_name, b)
            a_hash = _sha256(workdir / a_name)
            b_hash = _sha256(workdir / b_name)
            case_id = f"qmul-f32-seed-{seed}-n-{items}-{a_hash[:12]}-{b_hash[:12]}"
            case = {
                "case_id": case_id,
                "items": items,
                "iterations": iterations,
                "warmup": warmup,
                "samples": samples,
                "shape": [items, 4],
                "inputs": {
                    "a": a_name,
                    "b": b_name,
                    "a_sha256": a_hash,
                    "b_sha256": b_hash,
                },
                "outputs": {"out": out_name},
            }
            manifest_cases.append(case)
            prepared.append((case, a, b))

        manifest = {
            "schema": PERSISTENT_PROTOCOL,
            "workload": "qmul",
            "dtype": "float32",
            "lane_order": ["real", "i", "j", "k"],
            "input_format": "raw little-endian float32 row-major",
            "output_format": "raw little-endian float32 row-major",
            "seed": seed,
            "device_id": 0,
            "cases": manifest_cases,
            "outputs": {"metrics": "metrics.json"},
        }
        manifest_path = workdir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        host_started = time.perf_counter()
        _run_candidate(
            command,
            workdir=workdir,
            manifest_path=manifest_path,
            candidate_hash=candidate_hash,
            source_commit=source_commit,
        )
        host_process_s = time.perf_counter() - host_started
        metrics = json.loads((workdir / "metrics.json").read_text(encoding="utf-8"))
        normalized = validate_persistent_metrics(
            metrics,
            manifest,
            candidate_sha256=candidate_hash,
            host_process_s=host_process_s,
        )

        results: list[dict[str, object]] = []
        metric_cases = normalized["cases"]
        assert isinstance(metric_cases, list)
        for (case, a, b), case_metric in zip(prepared, metric_cases, strict=True):
            items = int(case["items"])
            output_path = workdir / str(case["outputs"]["out"])  # type: ignore[index]
            output = _read_float32(output_path, (items, 4))
            _, correctness = validate_qmul_output(output, a, b)
            if _fnv1a64(output_path.read_bytes()) != case_metric["output_identity"]["fnv1a64"]:  # type: ignore[index]
                raise IntegrityError(f"persistent output checksum mismatch for N={items}")
            if int(case_metric["output_identity"]["value_count"]) != items * 4:  # type: ignore[index]
                raise IntegrityError(f"persistent output value count mismatch for N={items}")

            samples = [float(value) for value in case_metric["timings_s"]["samples"]]  # type: ignore[index]
            device_timing = timing_summary(samples)
            elapsed_s = float(device_timing["median"])
            iterations = int(case["iterations"])
            diff = output.to(torch.float64) - torch.stack(_hamilton64(a, b), dim=-1)
            max_abs = float(diff.abs().max().item())
            rms = float(torch.sqrt(torch.mean(diff * diff)).item())
            denominator = torch.stack(_hamilton64(a, b), dim=-1).abs().clamp_min(1e-12)
            max_rel = float((diff.abs() / denominator).max().item())
            flops = items * iterations * 28
            total_bytes = items * iterations * 48
            results.append(
                {
                    "suite": "qmul",
                    "workload": "qmul",
                    "backend": "external-qmul",
                    "device": DEVICE,
                    "execution_label": "hardware",
                    "stable_benchmark": False,
                    "methodology_note": methodology_note,
                    "dtype": "float32",
                    "structured_shape": f"[{items}, 4]",
                    "items": items,
                    "iterations": iterations,
                    "warmup": int(case["warmup"]),
                    "elapsed_s": elapsed_s,
                    "latency_ms": elapsed_s / iterations * 1000.0,
                    "throughput": items * iterations / elapsed_s,
                    "throughput_unit": "qmul/s",
                    "max_abs_error": max_abs,
                    "max_rel_error": max_rel,
                    "rms_error": rms,
                    "checksum": float(output.to(torch.float64).sum().item()),
                    "output_sha256": _sha256(output_path),
                    "case_id": case["case_id"],
                    "correctness": correctness,
                    "scalar_reference_max_abs_error": correctness[
                        "scalar_first_eight_max_abs_error"
                    ],
                    "stability_max_abs": None,
                    "implementation_class": IMPLEMENTATION_CLASS,
                    "performance_eligible": True,
                    "candidate_metadata": case_metric["work"],
                    "timing": {
                        "primary_elapsed": "device_s.median",
                        "repetitions": len(samples),
                        "device_s": device_timing,
                        "phases_s": case_metric["timings_s"],
                    },
                    "estimated_flops": flops,
                    "estimated_flops_per_s": flops / elapsed_s,
                    "estimated_bytes_read": items * iterations * 32,
                    "estimated_bytes_written": items * iterations * 16,
                    "estimated_total_bytes": total_bytes,
                    "effective_gb_per_s": total_bytes / elapsed_s / 1e9,
                    "arithmetic_intensity_flops_per_byte": 28 / 48,
                    "provenance": metrics["provenance"],
                }
            )

    report: dict[str, object] = {
        "schema": "structuredbench.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "suite": "qmul",
        "backend": "external-qmul",
        "benchmark_stage": benchmark_stage,
        "measurement_mode": "persistent_device_session.v1",
        "protocol": PERSISTENT_PROTOCOL,
        "device": DEVICE,
        "execution_label": "hardware",
        "stable_benchmark": False,
        "methodology_note": methodology_note,
        "dtype": "float32",
        "seed": seed,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "external_command": command,
        "repetitions": 1 if benchmark_stage == "conformance" else PERFORMANCE_SAMPLES,
        "case_items": [spec[0] for spec in case_specs],
        "session_timing": {
            "host_process_end_to_end_s": host_process_s,
            **metrics["session_timings_s"],
        },
        "lifecycle": metrics["lifecycle"],
        "provenance": {
            "repository_commit": source_commit,
            "candidate_sha256": candidate_hash,
            "candidate": metrics["provenance"],
        },
        "results": results,
    }
    validate_persistent_report(report)
    return report


def validate_persistent_metrics(
    metrics: object,
    manifest: Mapping[str, object],
    *,
    candidate_sha256: str,
    host_process_s: float,
) -> dict[str, object]:
    """Strictly validate candidate metrics before accepting output evidence."""

    if not isinstance(metrics, dict):
        raise IntegrityError("persistent metrics must be an object")
    expected_top = {
        "schema": PERSISTENT_METRICS_SCHEMA,
        "protocol": PERSISTENT_PROTOCOL,
        "device": DEVICE,
        "dtype": "float32",
        "execution_kind": "hardware",
        "implementation_class": IMPLEMENTATION_CLASS,
        "performance_eligible": True,
        "stable_benchmark": False,
    }
    for key, expected in expected_top.items():
        if metrics.get(key) != expected:
            raise IntegrityError(
                f"persistent metrics {key} mismatch: expected {expected!r}, got {metrics.get(key)!r}"
            )
    lifecycle = metrics.get("lifecycle")
    if lifecycle != {
        "device_count": 1,
        "device_id": 0,
        "create_count": 1,
        "close_count": 1,
    }:
        raise IntegrityError("persistent metrics require exactly one device-0 create/close lifecycle")
    provenance = metrics.get("provenance")
    if not isinstance(provenance, dict):
        raise IntegrityError("persistent hardware metrics require provenance")
    required_provenance = (*HARDWARE_PROVENANCE_FIELDS, "candidate_sha256", "repository_commit")
    for key in required_provenance:
        value = provenance.get(key)
        if not isinstance(value, str) or not value.strip() or value.lower() in {
            "unknown",
            "none",
            "n/a",
            "unset",
        }:
            raise IntegrityError(f"persistent hardware metrics missing provenance.{key}")
    if provenance["candidate_sha256"] != candidate_sha256:
        raise IntegrityError("persistent candidate_sha256 does not match the executed binary")

    session = metrics.get("session_timings_s")
    if not isinstance(session, dict):
        raise IntegrityError("persistent metrics require session_timings_s")
    for key in ("device_create", "device_close", "candidate_session"):
        _finite_nonnegative(session.get(key), f"session_timings_s.{key}")
    candidate_session = float(session["candidate_session"])
    host_s = _finite_positive(host_process_s, "host process end-to-end")
    if candidate_session > host_s * 1.05 + 1e-6:
        raise IntegrityError("candidate session timing exceeds host process end-to-end timing")

    expected_cases = manifest.get("cases")
    actual_cases = metrics.get("cases")
    if not isinstance(expected_cases, list) or not isinstance(actual_cases, list):
        raise IntegrityError("persistent manifest and metrics require cases arrays")
    if len(actual_cases) != len(expected_cases):
        raise IntegrityError("persistent metrics case count mismatch")
    normalized_cases: list[dict[str, object]] = []
    accounted_s = float(session["device_create"]) + float(session["device_close"])
    for expected, actual in zip(expected_cases, actual_cases, strict=True):
        if not isinstance(expected, dict) or not isinstance(actual, dict):
            raise IntegrityError("persistent case must be an object")
        for key in ("case_id", "items", "iterations", "warmup", "samples"):
            if actual.get(key) != expected.get(key):
                raise IntegrityError(f"persistent case {key} mismatch")
        expected_input = expected.get("inputs")
        if not isinstance(expected_input, dict) or actual.get("input_identity") != {
            "a_sha256": expected_input.get("a_sha256"),
            "b_sha256": expected_input.get("b_sha256"),
        }:
            raise IntegrityError("persistent case input identity mismatch")
        output_identity = actual.get("output_identity")
        if not isinstance(output_identity, dict) or not _is_hex(
            output_identity.get("fnv1a64"), 16
        ):
            raise IntegrityError("persistent case requires a 64-bit output checksum")
        timings = actual.get("timings_s")
        if not isinstance(timings, dict):
            raise IntegrityError("persistent case requires timings_s")
        for key in (
            "buffer_allocation",
            "program_build",
            "h2d",
            "prewarm_sync",
            "warmup",
            "d2h",
            "cleanup",
        ):
            accounted_s += _finite_nonnegative(timings.get(key), f"timings_s.{key}")
        samples = timings.get("samples")
        if not isinstance(samples, list) or len(samples) != int(expected["samples"]):
            raise IntegrityError("persistent timing sample count mismatch")
        for value in samples:
            accounted_s += _finite_positive(value, "timings_s.samples[]")
        _validate_work(actual.get("work"), items=int(expected["items"]))
        normalized_cases.append(actual)
    if accounted_s > candidate_session * 1.05 + 1e-6:
        raise IntegrityError("persistent phase timings exceed candidate session timing")
    return {**metrics, "cases": normalized_cases}


def validate_persistent_report(report: Mapping[str, object]) -> None:
    """Validate additive persistent metadata plus the unchanged report contract."""

    if report.get("measurement_mode") != "persistent_device_session.v1":
        raise IntegrityError("persistent report measurement mode mismatch")
    if report.get("protocol") != PERSISTENT_PROTOCOL:
        raise IntegrityError("persistent report protocol mismatch")
    if report.get("stable_benchmark") is not False:
        raise IntegrityError("first persistent report must keep stable_benchmark=false")
    if report.get("lifecycle") != {
        "device_count": 1,
        "device_id": 0,
        "create_count": 1,
        "close_count": 1,
    }:
        raise IntegrityError("persistent report lifecycle mismatch")
    results = report.get("results")
    if not isinstance(results, list):
        raise IntegrityError("persistent report results must be a list")
    if {result.get("implementation_class") for result in results if isinstance(result, dict)} != {
        IMPLEMENTATION_CLASS
    }:
        raise IntegrityError("persistent report implementation class mismatch")
    validate_report(report)


def render_persistent_markdown(report: Mapping[str, object]) -> str:
    results = report["results"]
    assert isinstance(results, list)
    lines = [
        "# Persistent-device Stage B qmul report",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        f"Stage: `{report['benchmark_stage']}`",
        f"Device: `{report['device']}`",
        f"Implementation: `{IMPLEMENTATION_CLASS}`",
        "Performance eligible: `true`",
        "Stable benchmark: `false`",
        "",
        str(report["methodology_note"]),
        "",
        "## Validated results",
        "",
        "| N | values | iters | samples | median device s | p95 device s | max abs error |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        assert isinstance(result, dict)
        timing = result["timing"]["device_s"]  # type: ignore[index]
        lines.append(
            f"| {result['items']} | {result['correctness']['validated_values']} | "  # type: ignore[index]
            f"{result['iterations']} | {result['timing']['repetitions']} | "  # type: ignore[index]
            f"{float(timing['median']):.9f} | {float(timing['p95']):.9f} | "
            f"{float(result['max_abs_error']):.3e} |"
        )
    lines.extend(
        [
            "",
            "## Lifecycle",
            "",
            "One host process created Wormhole device 0 once, executed every listed case, and closed it once.",
            "Device 1 was not opened or used.",
            "",
            "## Timing contract",
            "",
            "The primary elapsed field remains prepared-workload device time. Additive phase records expose device creation, buffer allocation, program build, H2D, prewarm synchronization, warmup, each measured sample, D2H, cleanup, device close, and host process end-to-end time.",
            "",
            "This is a first persistent-device hardware sample. It is not a stability result, acceleration claim, or CPU comparison.",
            "",
        ]
    )
    return "\n".join(lines)


def _run_candidate(
    command: str,
    *,
    workdir: Path,
    manifest_path: Path,
    candidate_hash: str,
    source_commit: str,
) -> None:
    tokens = shlex.split(command)
    if not tokens:
        raise IntegrityError("persistent candidate command must not be empty")
    lowered = command.lower()
    if any(word in lowered for word in ("emule", "docker", "podman", "reference")):
        raise IntegrityError("persistent hardware command may not use emulation, containers, or references")
    env = os.environ.copy()
    env.update(
        {
            "TT_RQM_PERSISTENT_QMUL_DIR": str(workdir),
            "TT_RQM_PERSISTENT_QMUL_MANIFEST": str(manifest_path),
            "TT_RQM_EXECUTION_LABEL": "hardware",
            "TT_RQM_CANDIDATE_SHA256": candidate_hash,
            "TT_RQM_BUILD_ID": candidate_hash,
            "TT_RQM_REPOSITORY_COMMIT": source_commit,
        }
    )
    completed = subprocess.run(tokens, capture_output=True, text=True, env=env)
    if completed.returncode:
        raise IntegrityError(
            "persistent qmul candidate failed\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )


def _validate_work(value: object, *, items: int) -> None:
    if not isinstance(value, dict):
        raise IntegrityError("persistent case requires work metadata")
    tiles = (items + 1023) // 1024
    expected = {
        "device_count": 1,
        "device_id": 0,
        "component_tiles": tiles,
        "layout": "planar_float32_tiles_32x32",
        "work_split": "row_major",
        "arithmetic_path": "tensix_compute_sfpu",
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise IntegrityError(f"persistent work.{key} mismatch")
    for key in ("core_count", "grid_x", "grid_y", "available_core_count"):
        if not isinstance(value.get(key), int) or int(value[key]) <= 0:
            raise IntegrityError(f"persistent work.{key} must be positive")
    if value["grid_x"] * value["grid_y"] != value["available_core_count"]:
        raise IntegrityError("persistent work grid mismatch")
    if value["core_count"] != min(tiles, value["available_core_count"]):
        raise IntegrityError("persistent work core count mismatch")


def _write_float32(path: Path, tensor: torch.Tensor) -> None:
    values = array("f", tensor.contiguous().view(-1).tolist())
    if sys.byteorder != "little":
        values.byteswap()
    path.write_bytes(values.tobytes())


def _read_float32(path: Path, shape: tuple[int, int]) -> torch.Tensor:
    values = array("f")
    values.frombytes(path.read_bytes())
    if sys.byteorder != "little":
        values.byteswap()
    if len(values) != shape[0] * shape[1]:
        raise IntegrityError(f"persistent output length mismatch for {path.name}")
    return torch.tensor(values, dtype=torch.float32).reshape(shape)


def _hamilton64(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, ...]:
    ar, ai, aj, ak = a.to(torch.float64).unbind(-1)
    br, bi, bj, bk = b.to(torch.float64).unbind(-1)
    return (
        ar * br - ai * bi - aj * bj - ak * bk,
        ar * bi + ai * br + aj * bk - ak * bj,
        ar * bj - ai * bk + aj * br + ak * bi,
        ar * bk + ai * bj - aj * bi + ak * br,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fnv1a64(payload: bytes) -> str:
    value = 14695981039346656037
    for byte in payload:
        value ^= byte
        value = (value * 1099511628211) & ((1 << 64) - 1)
    return f"{value:016x}"


def _finite_nonnegative(value: object, name: str) -> float:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise IntegrityError(f"{name} must be numeric") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise IntegrityError(f"{name} must be nonnegative and finite")
    return parsed


def _finite_positive(value: object, name: str) -> float:
    parsed = _finite_nonnegative(value, name)
    if parsed <= 0:
        raise IntegrityError(f"{name} must be positive")
    return parsed


def _is_hex(value: object, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and all(
        character in "0123456789abcdef" for character in value
    )
