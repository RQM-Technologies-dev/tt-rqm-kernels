#!/usr/bin/env python3
"""CPU/PyTorch reference command for the external-qmul StructuredBench protocol."""

from __future__ import annotations

import argparse
from array import array
import json
import os
from pathlib import Path
import sys
import time

import torch

from tt_rqm_kernels.backends import torch_backend

PROTOCOL = "tt-rqm-external-qmul.v1"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the CPU/PyTorch external-qmul reference command."
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="External qmul work directory. Defaults to TT_RQM_EXTERNAL_QMUL_DIR.",
    )
    args = parser.parse_args()

    work_dir = args.workdir or _env_work_dir()
    manifest_path = Path(
        os.environ.get("TT_RQM_EXTERNAL_QMUL_MANIFEST", work_dir / "manifest.json")
    )
    manifest = _load_manifest(manifest_path)
    shape = _validate_manifest(manifest)
    iterations = int(manifest["iterations"])
    warmup = int(manifest["warmup"])

    a_path = work_dir / str(manifest["inputs"]["a"])
    b_path = work_dir / str(manifest["inputs"]["b"])
    out_path = work_dir / str(manifest["outputs"]["out"])
    metrics_path = work_dir / str(manifest["outputs"]["metrics"])

    a = _read_float32_binary(a_path, shape)
    b = _read_float32_binary(b_path, shape)

    output = None
    with torch.no_grad():
        for _ in range(warmup):
            output = torch_backend.qmul(a, b)
        start = time.perf_counter()
        for _ in range(iterations):
            output = torch_backend.qmul(a, b)
        elapsed_s = time.perf_counter() - start

    if output is None:
        output = torch_backend.qmul(a, b)

    _write_float32_binary(out_path, output)
    metrics_path.write_text(
        json.dumps(
            {
                "schema": "tt-rqm-external-qmul-metrics.v1",
                "protocol": PROTOCOL,
                "backend": "external-qmul-reference",
                "device": "cpu/pytorch-reference",
                "dtype": "float32",
                "items": int(manifest["items"]),
                "iterations": iterations,
                "warmup": warmup,
                "elapsed_s": elapsed_s,
                "note": (
                    "CPU/PyTorch reference command for validating the "
                    "external-qmul harness; not a hardware performance result."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


def _env_work_dir() -> Path:
    value = os.environ.get("TT_RQM_EXTERNAL_QMUL_DIR")
    if not value:
        raise SystemExit("TT_RQM_EXTERNAL_QMUL_DIR is required unless --workdir is set")
    return Path(value)


def _load_manifest(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("manifest.json must contain a JSON object")
    return payload


def _validate_manifest(manifest: dict[str, object]) -> tuple[int, int]:
    if manifest.get("schema") != PROTOCOL:
        raise ValueError(f"unsupported protocol: {manifest.get('schema')!r}")
    if manifest.get("workload") != "qmul":
        raise ValueError(f"unsupported workload: {manifest.get('workload')!r}")
    if manifest.get("dtype") != "float32":
        raise ValueError(f"unsupported dtype: {manifest.get('dtype')!r}")

    shape = manifest.get("shape")
    items = int(manifest["items"])
    if shape != [items, 4]:
        raise ValueError(f"expected shape [{items}, 4], got {shape!r}")
    if int(manifest["iterations"]) <= 0:
        raise ValueError("iterations must be positive")
    if int(manifest["warmup"]) < 0:
        raise ValueError("warmup must be nonnegative")
    return items, 4


def _read_float32_binary(path: Path, shape: tuple[int, int]) -> torch.Tensor:
    payload = array("f")
    payload.frombytes(path.read_bytes())
    if sys.byteorder != "little":
        payload.byteswap()
    expected_values = shape[0] * shape[1]
    if len(payload) != expected_values:
        raise ValueError(
            f"{path.name} has {len(payload)} float32 values; expected {expected_values}"
        )
    return torch.tensor(payload, dtype=torch.float32).reshape(shape)


def _write_float32_binary(path: Path, value: torch.Tensor) -> None:
    flat = value.detach().cpu().to(dtype=torch.float32).contiguous().reshape(-1)
    payload = array("f", flat.tolist())
    if sys.byteorder != "little":
        payload.byteswap()
    path.write_bytes(payload.tobytes())


if __name__ == "__main__":
    raise SystemExit(main())
