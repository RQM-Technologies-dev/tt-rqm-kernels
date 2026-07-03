#!/usr/bin/env python3
"""Small no-torch qmul candidate used by tests."""

from __future__ import annotations

from array import array
import json
import os
from pathlib import Path
import sys
import time


def main() -> int:
    work_dir = Path(os.environ["TT_RQM_EXTERNAL_QMUL_DIR"])
    manifest = json.loads(Path(os.environ["TT_RQM_EXTERNAL_QMUL_MANIFEST"]).read_text())
    items = int(manifest["items"])
    iterations = int(manifest["iterations"])
    warmup = int(manifest["warmup"])
    a = _read(work_dir / "a.bin")
    b = _read(work_dir / "b.bin")

    out: array[float] | None = None
    for _ in range(warmup):
        out = _qmul(a, b, items)
    start = time.perf_counter()
    for _ in range(iterations):
        out = _qmul(a, b, items)
    elapsed_s = time.perf_counter() - start
    if out is None:
        out = _qmul(a, b, items)

    _write(work_dir / "out.bin", out)
    (work_dir / "metrics.json").write_text(
        json.dumps(
            {
                "elapsed_s": elapsed_s,
                "device": "cpu/python-test-fixture",
            },
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
