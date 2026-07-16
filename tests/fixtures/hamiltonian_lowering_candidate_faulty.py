#!/usr/bin/env python3
"""No-Torch H2A protocol fixture with optional output fault injection."""

from __future__ import annotations

from array import array
import json
import math
import os
from pathlib import Path
import sys
import time


def main() -> int:
    work_dir = Path(os.environ["TT_RQM_H2A_DIR"])
    manifest = json.loads(Path(os.environ["TT_RQM_H2A_MANIFEST"]).read_text())
    coefficients = _read(work_dir / manifest["inputs"]["hamiltonians"])
    dt = _read(work_dir / manifest["inputs"]["dt"])
    B, K, _ = manifest["hamiltonian_shape"]
    started = time.perf_counter()
    rotors, phases = _lower(coefficients, dt, B=B, K=K, hbar=float(manifest["hbar"]))
    elapsed = time.perf_counter() - started
    rotor_path = work_dir / manifest["outputs"]["rotors"]
    phase_path = work_dir / manifest["outputs"]["phases"]
    _write(rotor_path, rotors)
    _write(phase_path, phases)
    metrics = {
        "schema": "tt-rqm-external-hamiltonian-lowering-metrics.v1",
        "protocol": manifest["schema"],
        "benchmark": manifest["benchmark"],
        "stage": manifest["stage"],
        "dtype": "float32",
        "execution_label": "cpu_reference",
        "hamiltonian_shape": manifest["hamiltonian_shape"],
        "dt_shape": manifest["dt_shape"],
        "stable_benchmark": False,
        "performance_eligible": False,
        "candidate_metadata": {
            "implementation_class": "cpu_reference",
            "device": "cpu/pytorch-reference",
        },
        "timings_s": {"candidate": elapsed},
    }
    metrics_path = work_dir / manifest["outputs"]["metrics"]
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    fault = os.environ.get("TT_RQM_H2A_TEST_FAULT")
    if fault == "truncate":
        rotor_path.write_bytes(rotor_path.read_bytes()[:-4])
    elif fault == "nonfinite":
        values = _read(rotor_path)
        values[-1] = math.nan
        _write(rotor_path, values)
    elif fault == "reorder":
        values = _read(rotor_path)
        first = values[0:4]
        third = values[8:12]
        values[0:4] = third
        values[8:12] = first
        _write(rotor_path, values)
    elif fault == "metrics":
        metrics["stage"] = "wrong"
        metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    elif fault == "missing":
        rotor_path.unlink()
    return 0


def _lower(
    coefficients: array[float], dt: array[float], *, B: int, K: int, hbar: float
) -> tuple[array[float], array[float]]:
    rotors = array("f")
    phases = array("f")
    for index in range(B * K):
        h0, hx, hy, hz = coefficients[index * 4 : index * 4 + 4]
        step = dt[0 if len(dt) == 1 else index] / hbar
        magnitude = math.sqrt(hx * hx + hy * hy + hz * hz)
        angle = magnitude * step
        if magnitude == 0.0:
            rotors.extend((1.0, 0.0, 0.0, 0.0))
        else:
            scale = math.sin(angle) / magnitude
            rotors.extend((math.cos(angle), hx * scale, hy * scale, hz * scale))
        alpha = h0 * step
        phases.extend((math.cos(alpha), -math.sin(alpha)))
    return rotors, phases


def _read(path: Path) -> array[float]:
    values = array("f")
    values.frombytes(path.read_bytes())
    if sys.byteorder != "little":
        values.byteswap()
    return values


def _write(path: Path, values: array[float]) -> None:
    payload = array("f", values)
    if sys.byteorder != "little":
        payload.byteswap()
    path.write_bytes(payload.tobytes())


if __name__ == "__main__":
    raise SystemExit(main())
