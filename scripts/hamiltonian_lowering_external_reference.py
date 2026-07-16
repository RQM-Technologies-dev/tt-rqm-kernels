#!/usr/bin/env python3
"""CPU/PyTorch implementation of the external H2A candidate protocol."""

from __future__ import annotations

from array import array
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch

from tt_rqm_kernels.hamiltonian.su2_lowering import lower_two_level_hamiltonian
from tt_rqm_kernels.hamiltonian_lowering_candidate import METRICS_SCHEMA, PROTOCOL


def main() -> int:
    work_dir = Path(os.environ["TT_RQM_H2A_DIR"])
    manifest = json.loads(Path(os.environ["TT_RQM_H2A_MANIFEST"]).read_text())
    _validate_manifest(manifest, work_dir)
    shape = tuple(manifest["hamiltonian_shape"])
    dt_shape = tuple(manifest["dt_shape"])
    setup_started = time.perf_counter()
    coefficients = _read(work_dir / manifest["inputs"]["hamiltonians"], shape)
    dt = _read(work_dir / manifest["inputs"]["dt"], dt_shape)
    setup_s = time.perf_counter() - setup_started
    started = time.perf_counter()
    rotors, phases = lower_two_level_hamiltonian(coefficients, dt, hbar=float(manifest["hbar"]))
    device_s = time.perf_counter() - started
    _write(work_dir / manifest["outputs"]["rotors"], rotors)
    _write(work_dir / manifest["outputs"]["phases"], phases)
    metrics = {
        "schema": METRICS_SCHEMA,
        "protocol": PROTOCOL,
        "benchmark": "HamiltonianLoweringBench",
        "stage": manifest["stage"],
        "dtype": "float32",
        "execution_label": "cpu_reference",
        "hamiltonian_shape": list(shape),
        "dt_shape": list(dt_shape),
        "stable_benchmark": False,
        "performance_eligible": False,
        "candidate_metadata": {
            "implementation_class": "cpu_reference",
            "device": "cpu/pytorch-reference",
        },
        "timings_s": {"setup": setup_s, "candidate": device_s},
    }
    (work_dir / manifest["outputs"]["metrics"]).write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


def _validate_manifest(manifest: dict[str, object], work_dir: Path) -> None:
    if manifest.get("schema") != PROTOCOL:
        raise ValueError("unsupported H2A protocol")
    if manifest.get("benchmark") != "HamiltonianLoweringBench":
        raise ValueError("unsupported benchmark")
    if manifest.get("dtype") != "float32":
        raise ValueError("H2A protocol requires float32")
    inputs = manifest["inputs"]
    assert isinstance(inputs, dict)
    for name in ("hamiltonians", "dt"):
        path = work_dir / str(inputs[name])
        expected = inputs[f"{name}_sha256"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"{name} input hash mismatch")


def _read(path: Path, shape: tuple[int, ...]) -> torch.Tensor:
    values = array("f")
    values.frombytes(path.read_bytes())
    if sys.byteorder != "little":
        values.byteswap()
    expected = 1
    for dimension in shape:
        expected *= dimension
    if len(values) != expected:
        raise ValueError(f"{path.name} size mismatch")
    return torch.tensor(values, dtype=torch.float32).reshape(shape)


def _write(path: Path, value: torch.Tensor) -> None:
    payload = array("f", value.detach().cpu().contiguous().reshape(-1).tolist())
    if sys.byteorder != "little":
        payload.byteswap()
    path.write_bytes(payload.tobytes())


if __name__ == "__main__":
    raise SystemExit(main())
