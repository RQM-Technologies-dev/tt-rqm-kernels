"""Python adapter for Tenstorrent external-qmul candidates."""

from __future__ import annotations

from array import array
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time

import torch

from tt_rqm_kernels.backends import torch_backend
from tt_rqm_kernels.backends.tenstorrent.availability import (
    Mode,
    resolve_execution_path,
)
from tt_rqm_kernels.backends.tenstorrent.report import (
    ExecutionLabel,
    methodology_note_for_label,
    validate_external_qmul_label,
    validate_stable_benchmark,
)
from tt_rqm_kernels.structuredbench import EXTERNAL_QMUL_PROTOCOL, run_suite


class TenstorrentAdapterError(RuntimeError):
    """Raised when a Tenstorrent external-qmul adapter cannot run."""


@dataclass(frozen=True)
class ExternalQmulRun:
    """One direct external-qmul run on caller-provided tensors."""

    output: torch.Tensor
    reference: torch.Tensor
    max_abs_error: float
    rms_error: float
    checksum: float
    elapsed_s: float
    latency_ms: float
    throughput: float
    device: str
    metrics: dict[str, object]


def run_configured_qmul(
    mode: Mode,
    *,
    command: str | None = None,
    items: int = 128,
    iterations: int = 1,
    warmup: int = 0,
    seed: int = 0,
    stable_benchmark: bool = False,
    methodology_note: str | None = None,
) -> dict[str, object]:
    """Run the StructuredBench qmul suite through a configured Tenstorrent path."""

    path = resolve_execution_path(mode, command=command)
    if not path.available or not path.command:
        raise TenstorrentAdapterError(path.reason)
    if stable_benchmark and path.execution_label == "emulation":
        raise TenstorrentAdapterError(
            "stable benchmark reports are not allowed for --mode emule; "
            "tt-emule output must use stable_benchmark=false"
        )
    return run_structuredbench_qmul(
        command=path.command,
        execution_label=path.execution_label,
        items=items,
        iterations=iterations,
        warmup=warmup,
        seed=seed,
        stable_benchmark=stable_benchmark,
        methodology_note=methodology_note,
    )


def run_structuredbench_qmul(
    *,
    command: str,
    execution_label: ExecutionLabel,
    items: int = 128,
    iterations: int = 1,
    warmup: int = 0,
    seed: int = 0,
    stable_benchmark: bool = False,
    methodology_note: str | None = None,
) -> dict[str, object]:
    """Run StructuredBench qmul through an external Tenstorrent candidate."""

    label = validate_external_qmul_label(execution_label, command=command)
    validate_stable_benchmark(label, stable_benchmark=stable_benchmark)
    return run_suite(
        "qmul",
        backend="external-qmul",
        dtype_name="float32",
        seed=seed,
        items_override=items,
        iterations_override=iterations,
        warmup_override=warmup,
        external_command=command,
        execution_label=label,
        stable_benchmark=stable_benchmark,
        methodology_note=methodology_note
        or methodology_note_for_label(label, stable_benchmark=stable_benchmark),
    )


def run_external_qmul_inputs(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    command: str | None,
    iterations: int = 1,
    warmup: int = 0,
    seed: int = 0,
) -> ExternalQmulRun:
    """Run a command implementing the external-qmul protocol on explicit inputs."""

    if not command:
        raise TenstorrentAdapterError(
            "external qmul command is not configured; pass --command or use a "
            "configured emule/hardware mode"
        )
    a32 = _as_float32_qtensor(a, "a")
    b32 = _as_float32_qtensor(b, "b")
    if a32.shape != b32.shape:
        raise TenstorrentAdapterError(
            f"a and b must have the same shape, got {tuple(a32.shape)} and {tuple(b32.shape)}"
        )
    items = int(a32.shape[0])
    if iterations <= 0:
        raise TenstorrentAdapterError("iterations must be positive")
    if warmup < 0:
        raise TenstorrentAdapterError("warmup must be nonnegative")

    reference = torch_backend.qmul(
        a32.to(dtype=torch.float64),
        b32.to(dtype=torch.float64),
    ).to(dtype=torch.float32)

    with tempfile.TemporaryDirectory(prefix="tt-rqm-direct-qmul-") as tmp_dir:
        work_dir = Path(tmp_dir)
        manifest_path = work_dir / "manifest.json"
        _write_float32_binary(work_dir / "a.bin", a32)
        _write_float32_binary(work_dir / "b.bin", b32)
        manifest_path.write_text(
            json.dumps(
                {
                    "schema": EXTERNAL_QMUL_PROTOCOL,
                    "workload": "qmul",
                    "dtype": "float32",
                    "lane_order": ["real", "i", "j", "k"],
                    "items": items,
                    "iterations": iterations,
                    "warmup": warmup,
                    "shape": [items, 4],
                    "input_format": "raw little-endian float32 row-major",
                    "output_format": "raw little-endian float32 row-major",
                    "inputs": {"a": "a.bin", "b": "b.bin"},
                    "outputs": {"out": "out.bin", "metrics": "metrics.json"},
                    "seed": seed,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        _run_external_command(command, work_dir=work_dir, manifest_path=manifest_path)
        metrics = _load_metrics(work_dir / "metrics.json")
        output = _read_float32_binary(work_dir / "out.bin", (items, 4))

    elapsed_s = float(metrics.get("elapsed_s", 0.0))
    if elapsed_s <= 0.0:
        # Keep direct example output usable if a candidate only writes out.bin.
        elapsed_s = float(metrics.get("latency_s", 0.0)) or 1e-12
    diff = output.to(dtype=torch.float64) - reference.to(dtype=torch.float64)
    max_abs_error = float(diff.abs().max().item())
    rms_error = float(torch.sqrt(torch.mean(diff * diff)).item())
    checksum = float(output.to(dtype=torch.float64).sum().item())
    return ExternalQmulRun(
        output=output,
        reference=reference,
        max_abs_error=max_abs_error,
        rms_error=rms_error,
        checksum=checksum,
        elapsed_s=elapsed_s,
        latency_ms=(elapsed_s / iterations) * 1000.0,
        throughput=(items * iterations) / elapsed_s,
        device=str(metrics.get("device", "external-command")),
        metrics=metrics,
    )


def _as_float32_qtensor(value: torch.Tensor, name: str) -> torch.Tensor:
    if value.ndim != 2 or value.shape[-1] != 4:
        raise TenstorrentAdapterError(f"{name} must have shape [N, 4]")
    return value.detach().cpu().to(dtype=torch.float32).contiguous()


def _run_external_command(command_text: str, *, work_dir: Path, manifest_path: Path) -> None:
    command = shlex.split(command_text)
    if not command:
        raise TenstorrentAdapterError("external qmul command must not be empty")
    env = os.environ.copy()
    env["TT_RQM_EXTERNAL_QMUL_DIR"] = str(work_dir)
    env["TT_RQM_EXTERNAL_QMUL_MANIFEST"] = str(manifest_path)
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True, env=env)
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise TenstorrentAdapterError(
            "external qmul command failed\n"
            f"command: {' '.join(command)}\n"
            f"work_dir: {work_dir}\n"
            f"elapsed_s: {elapsed:.6f}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )


def _load_metrics(path: Path) -> dict[str, object]:
    if not path.exists():
        raise TenstorrentAdapterError("external qmul command did not write metrics.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TenstorrentAdapterError("metrics.json must contain a JSON object")
    return payload


def _write_float32_binary(path: Path, value: torch.Tensor) -> None:
    flat = value.detach().cpu().to(dtype=torch.float32).contiguous().reshape(-1)
    payload = array("f", flat.tolist())
    if sys.byteorder != "little":
        payload.byteswap()
    path.write_bytes(payload.tobytes())


def _read_float32_binary(path: Path, shape: tuple[int, int]) -> torch.Tensor:
    if not path.exists():
        raise TenstorrentAdapterError("external qmul command did not write out.bin")
    values = array("f")
    values.frombytes(path.read_bytes())
    if sys.byteorder != "little":
        values.byteswap()
    expected = shape[0] * shape[1]
    if len(values) != expected:
        raise TenstorrentAdapterError(
            f"out.bin has {len(values)} float32 values; expected {expected}"
        )
    return torch.tensor(values, dtype=torch.float32).reshape(shape)
