"""Shared correctness, labeling, timing, and provenance gates for StructuredBench."""

from __future__ import annotations

import hashlib
import math
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Literal, Mapping, Sequence

import torch

from tt_rqm_kernels.backends import scalar_reference

EXTERNAL_METRICS_SCHEMA = "tt-rqm-external-qmul-metrics.v2"
QMUL_ATOL = 1e-4
QMUL_RTOL = 1e-4
ExecutionLabel = Literal["cpu", "simulator", "emulation", "hardware"]
BenchmarkStage = Literal["conformance", "performance"]
HARDWARE_PROVENANCE_FIELDS = (
    "chip_type",
    "tt_metal_commit",
    "compiler_version",
    "runtime_version",
    "build_id",
    "timer_scope",
)


class IntegrityError(ValueError):
    """Raised when benchmark evidence does not satisfy the shared contract."""


def validate_execution_policy(
    *,
    backend: str,
    execution_label: str,
    stable_benchmark: bool,
    command: str | None = None,
    stage: str | None = None,
    repetitions: int = 1,
    items: Sequence[int] = (),
    iterations: Sequence[int] = (),
    warmups: Sequence[int] = (),
) -> ExecutionLabel:
    if execution_label not in {"cpu", "simulator", "emulation", "hardware"}:
        raise IntegrityError(f"unsupported execution label: {execution_label}")
    if backend == "torch" and execution_label != "cpu":
        raise IntegrityError("torch backend reports must use execution_label=cpu")
    if backend == "tt-lang-sim" and execution_label != "simulator":
        raise IntegrityError("TT-Lang simulator reports must use execution_label=simulator")
    if backend == "external-qmul" and execution_label == "simulator":
        raise IntegrityError(
            "external-qmul reports should use cpu, emulation, or hardware; "
            "use tt-lang-sim for simulator reports"
        )
    validate_stability(execution_label, stable_benchmark=stable_benchmark)
    if execution_label == "hardware":
        validate_label_command(execution_label, command=command)
        if stage not in {"conformance", "performance"}:
            raise IntegrityError("hardware reports require an explicit benchmark stage")
    if stage == "conformance":
        if (
            list(items) != [128]
            or (iterations and list(iterations) != [1])
            or (warmups and list(warmups) != [0])
            or repetitions != 1
            or stable_benchmark
        ):
            raise IntegrityError(
                "conformance stage requires items=[128], iterations=[1], "
                "warmups=[0], repetitions=1, and stable_benchmark=false"
            )
    if stage == "performance":
        if execution_label != "hardware":
            raise IntegrityError("performance stage is only allowed on real hardware")
        if (
            list(items) != [4096, 65536, 262144]
            or (iterations and list(iterations) != [30, 30, 30])
            or (warmups and list(warmups) != [5, 5, 5])
            or repetitions < 10
        ):
            raise IntegrityError(
                "performance stage requires the 4096/65536/262144 sweep, "
                "30 iterations, 5 warmups, and at least 10 repetitions"
            )
    return execution_label  # type: ignore[return-value]


def validate_label_command(execution_label: str, *, command: str | None) -> None:
    if execution_label == "simulator":
        raise IntegrityError(
            "external-qmul reports should use cpu, emulation, or hardware; "
            "use tt-lang-sim for simulator reports"
        )
    if execution_label == "hardware":
        _reject_nonhardware_command(command)


def validate_stability(execution_label: str, *, stable_benchmark: bool) -> None:
    if stable_benchmark and execution_label != "hardware":
        raise IntegrityError(
            "stable benchmark reports are only allowed for real hardware runs"
        )


def independent_qmul_golden(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute Hamilton products in float64 from the exact float32 inputs."""

    a32 = a.detach().cpu().to(torch.float32).contiguous()
    b32 = b.detach().cpu().to(torch.float32).contiguous()
    if a32.shape != b32.shape or a32.ndim != 2 or a32.shape[1] != 4:
        raise IntegrityError("qmul inputs must have matching [N, 4] shapes")
    if not torch.isfinite(a32).all() or not torch.isfinite(b32).all():
        raise IntegrityError("qmul inputs contain non-finite values")
    a64 = a32.to(torch.float64)
    b64 = b32.to(torch.float64)
    ar, ai, aj, ak = a64.unbind(dim=-1)
    br, bi, bj, bk = b64.unbind(dim=-1)
    return torch.stack(
        (
            ar * br - ai * bi - aj * bj - ak * bk,
            ar * bi + ai * br + aj * bk - ak * bj,
            ar * bj - ai * bk + aj * br + ak * bi,
            ar * bk + ai * bj - aj * bi + ak * br,
        ),
        dim=-1,
    )


def validate_qmul_output(
    output: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    atol: float = QMUL_ATOL,
    rtol: float = QMUL_RTOL,
) -> tuple[torch.Tensor, dict[str, object]]:
    golden = independent_qmul_golden(a, b)
    actual = output.detach().cpu().to(torch.float64)
    if actual.shape != golden.shape:
        raise IntegrityError(
            f"qmul output shape {tuple(actual.shape)} does not match {tuple(golden.shape)}"
        )
    nonfinite = int((~torch.isfinite(actual)).sum().item())
    if nonfinite:
        raise IntegrityError(f"qmul output contains {nonfinite} non-finite values")
    diff = torch.abs(actual - golden)
    allowed = atol + rtol * torch.abs(golden)
    failing = int((diff > allowed).sum().item())
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    if failing:
        raise IntegrityError(
            f"qmul whole-output validation failed: {failing} values exceed "
            f"atol={atol} rtol={rtol}; max_abs_error={max_abs:.6e}"
        )
    sample_count = min(8, actual.shape[0])
    if sample_count:
        a64 = a.detach().cpu().to(torch.float32).to(torch.float64)
        b64 = b.detach().cpu().to(torch.float32).to(torch.float64)
        scalar_expected = torch.tensor(
            [
                scalar_reference.qmul_scalar(a64[index].tolist(), b64[index].tolist())
                for index in range(sample_count)
            ],
            dtype=torch.float64,
        )
        scalar_diagnostic = float(
            torch.abs(actual[:sample_count] - scalar_expected).max().item()
        )
    else:
        scalar_diagnostic = 0.0
    return golden, {
        "passed": True,
        "atol": atol,
        "rtol": rtol,
        "failing_values": 0,
        "nonfinite_values": 0,
        "validated_values": actual.numel(),
        "whole_output_max_abs_error": max_abs,
        "scalar_first_eight_max_abs_error": scalar_diagnostic,
        "golden": "independent Hamilton product; exact float32 inputs promoted to float64",
    }


def validate_external_metrics(
    metrics: Mapping[str, object],
    manifest: Mapping[str, object],
    *,
    execution_label: ExecutionLabel,
    host_end_to_end_s: float,
    candidate_sha256: str,
    stage: str | None,
) -> dict[str, object]:
    required_matches = {
        "schema": EXTERNAL_METRICS_SCHEMA,
        "protocol": manifest["schema"],
        "dtype": manifest["dtype"],
        "items": manifest["items"],
        "iterations": manifest["iterations"],
        "warmup": manifest["warmup"],
        "execution_kind": execution_label,
    }
    for key, expected in required_matches.items():
        if metrics.get(key) != expected:
            raise IntegrityError(
                f"external-qmul metrics {key} mismatch: expected {expected!r}, "
                f"got {metrics.get(key)!r}"
            )
    for key in ("backend", "device", "implementation_class"):
        if not isinstance(metrics.get(key), str) or not str(metrics[key]).strip():
            raise IntegrityError(f"external-qmul metrics require non-empty {key}")
    if not isinstance(metrics.get("performance_eligible"), bool):
        raise IntegrityError("external-qmul metrics require boolean performance_eligible")
    timings = metrics.get("timings_s")
    if not isinstance(timings, Mapping):
        raise IntegrityError("external-qmul metrics require timings_s object")
    setup_s = _positive_finite(timings.get("setup"), "timings_s.setup", allow_zero=True)
    device_s = _positive_finite(timings.get("device"), "timings_s.device")
    host_s = _positive_finite(host_end_to_end_s, "host end-to-end time")
    if setup_s + device_s > host_s * 1.05 + 1e-6:
        raise IntegrityError(
            "external-qmul setup plus device time exceeds host end-to-end time"
        )
    if stage == "performance" and metrics.get("performance_eligible") is not True:
        raise IntegrityError("performance stage requires performance_eligible=true")
    if execution_label == "hardware":
        provenance = metrics.get("provenance")
        if not isinstance(provenance, Mapping):
            raise IntegrityError("hardware metrics require provenance object")
        missing = [
            key
            for key in HARDWARE_PROVENANCE_FIELDS
            if (
                not isinstance(provenance.get(key), str)
                or not str(provenance[key]).strip()
                or str(provenance[key]).strip().lower()
                in {"unknown", "n/a", "none", "unset"}
            )
        ]
        if missing:
            raise IntegrityError(
                "hardware metrics missing provenance: " + ", ".join(missing)
            )
        if provenance.get("candidate_sha256") not in {None, candidate_sha256}:
            raise IntegrityError("hardware candidate_sha256 does not match observed command")
    return {
        "setup_s": setup_s,
        "device_s": device_s,
        "end_to_end_s": host_s,
        "candidate_sha256": candidate_sha256,
    }


def timing_summary(samples: Sequence[float]) -> dict[str, object]:
    if not samples:
        raise IntegrityError("timing samples must not be empty")
    values = sorted(_positive_finite(value, "timing sample", allow_zero=True) for value in samples)
    middle = len(values) // 2
    median = values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2
    p95_index = max(0, math.ceil(0.95 * len(values)) - 1)
    return {"samples": list(samples), "median": median, "p95": values[p95_index]}


def repository_commit(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def command_sha256(command: str, repo_root: Path) -> str:
    tokens = shlex.split(command)
    interpreter = Path(tokens[0]).name if tokens else ""
    candidates = tokens[1:] if interpreter.startswith(("python", "bash", "sh")) else tokens[:1]
    for token in candidates:
        path = Path(token)
        if not path.is_absolute():
            path = repo_root / path
        if path.is_file():
            return hashlib.sha256(path.read_bytes()).hexdigest()
    if tokens:
        executable = shutil.which(tokens[0])
        if executable and Path(executable).is_file():
            return hashlib.sha256(Path(executable).read_bytes()).hexdigest()
    return hashlib.sha256(command.encode("utf-8")).hexdigest()


def validate_report(report: Mapping[str, object]) -> None:
    label = str(report.get("execution_label"))
    stable = bool(report.get("stable_benchmark", False))
    validate_execution_policy(
        backend=str(report.get("backend")),
        execution_label=label,
        stable_benchmark=stable,
        command=str(report.get("external_command")) if report.get("external_command") else None,
        stage=str(report.get("benchmark_stage")) if report.get("benchmark_stage") else None,
        repetitions=int(report.get("repetitions", 1)),
        items=[int(item) for item in report.get("case_items", [])],
        iterations=[
            int(result.get("iterations", 0))
            for result in report.get("results", [])
            if isinstance(result, Mapping)
        ],
        warmups=[
            int(result.get("warmup", 0))
            for result in report.get("results", [])
            if isinstance(result, Mapping)
        ],
    )
    results = report.get("results")
    if not isinstance(results, list) or not results:
        raise IntegrityError("report results must be a non-empty list")
    for result in results:
        if not isinstance(result, Mapping):
            raise IntegrityError("report result must be an object")
        correctness = result.get("correctness")
        if not isinstance(correctness, Mapping) or correctness.get("passed") is not True:
            raise IntegrityError("every report result must pass correctness validation")
        for key in ("elapsed_s", "latency_ms", "throughput", "max_abs_error", "rms_error"):
            _positive_finite(result.get(key), f"result.{key}", allow_zero=key in {"max_abs_error", "rms_error"})


def _positive_finite(value: object, name: str, *, allow_zero: bool = False) -> float:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise IntegrityError(f"{name} must be numeric") from exc
    if not math.isfinite(parsed) or parsed < 0 or (parsed == 0 and not allow_zero):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise IntegrityError(f"{name} must be a {qualifier} finite value")
    return parsed


def _reject_nonhardware_command(command: str | None) -> None:
    if not command:
        raise IntegrityError("hardware reports require a configured command")
    lowered = command.lower()
    if "tt-emule" in lowered or "emule" in lowered or "run_candidate_docker" in lowered:
        raise IntegrityError(
            "hardware reports require a real Tenstorrent hardware command; "
            "tt-emule commands must use execution_label=emulation"
        )
    if "docker" in lowered or "podman" in lowered:
        raise IntegrityError(
            "hardware reports require a real Tenstorrent hardware command; "
            "Docker/container commands must not be labeled as hardware"
        )
    if "tt-lang-sim" in lowered or "qmul_external_reference" in lowered:
        raise IntegrityError(
            "hardware reports require a real Tenstorrent hardware command; "
            "simulator and CPU reference commands are rejected"
        )
