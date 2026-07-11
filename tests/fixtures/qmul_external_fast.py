#!/usr/bin/env python3
"""Small no-torch qmul candidate used by tests."""

from __future__ import annotations

from array import array
import json
import os
from pathlib import Path
import sys
import time
import math


def main() -> int:
    work_dir = Path(os.environ["TT_RQM_EXTERNAL_QMUL_DIR"])
    manifest = json.loads(Path(os.environ["TT_RQM_EXTERNAL_QMUL_MANIFEST"]).read_text())
    items = int(manifest["items"])
    iterations = int(manifest["iterations"])
    warmup = int(manifest["warmup"])
    setup_start = time.perf_counter()
    a = _read(work_dir / "a.bin")
    b = _read(work_dir / "b.bin")
    setup_s = time.perf_counter() - setup_start

    out: array[float] | None = None
    for _ in range(warmup):
        out = _qmul(a, b, items)
    start = time.perf_counter()
    for _ in range(iterations):
        out = _qmul(a, b, items)
    elapsed_s = time.perf_counter() - start
    if out is None:
        out = _qmul(a, b, items)
    if os.environ.get("TT_RQM_TEST_CORRUPT_AFTER_EIGHT") and items > 8:
        out[8 * 4] += 1.0
    if os.environ.get("TT_RQM_TEST_NAN_OUTPUT"):
        out[-1] = math.nan

    _write(work_dir / "out.bin", out)
    metrics = {
        "schema": "tt-rqm-external-qmul-metrics.v2",
        "protocol": manifest["schema"],
        "backend": "external-qmul-test-fixture",
        "device": "cpu/python-test-fixture",
        "dtype": "float32",
        "items": items,
        "iterations": iterations,
        "warmup": warmup,
        "execution_kind": os.environ.get("TT_RQM_EXECUTION_LABEL", "cpu"),
        "implementation_class": "cpu_test_fixture",
        "performance_eligible": False,
        "timings_s": {"setup": setup_s, "device": elapsed_s},
    }
    if os.environ.get("TT_RQM_TEST_MISMATCH_METRICS"):
        metrics["items"] = items + 1
    if os.environ.get("TT_RQM_TEST_FABRICATED_TIMING"):
        metrics["timings_s"] = {"setup": 1000.0, "device": 1000.0}
    (work_dir / "metrics.json").write_text(
        json.dumps(
            metrics,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


def _qmul(a: array[float], b: array[float], items: int) -> array[float]:
    out = array("f", [0.0]) * (items * 4)
    for index in range(items):
        base = index * 4
        ar, ai, aj, ak = a[base : base + 4]
        br, bi, bj, bk = b[base : base + 4]
        out[base] = ar * br - ai * bi - aj * bj - ak * bk
        out[base + 1] = ar * bi + ai * br + aj * bk - ak * bj
        out[base + 2] = ar * bj - ai * bk + aj * br + ak * bi
        out[base + 3] = ar * bk + ai * bj - aj * bi + ak * br
    return out


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
