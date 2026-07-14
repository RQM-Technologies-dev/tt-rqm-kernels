#!/usr/bin/env python3
"""Hardware-independent fixture for the persistent qmul wire contract."""

from __future__ import annotations

from array import array
import json
import os
from pathlib import Path
import sys


def read_f32(path: Path) -> array:
    values = array("f")
    values.frombytes(path.read_bytes())
    if sys.byteorder != "little":
        values.byteswap()
    return values


def write_f32(path: Path, values: array) -> None:
    output = array("f", values)
    if sys.byteorder != "little":
        output.byteswap()
    path.write_bytes(output.tobytes())


def fnv1a64(payload: bytes) -> str:
    value = 14695981039346656037
    for byte in payload:
        value ^= byte
        value = (value * 1099511628211) & ((1 << 64) - 1)
    return f"{value:016x}"


def main() -> int:
    device_id = 0
    if "--device" in sys.argv:
        device_id = int(sys.argv[sys.argv.index("--device") + 1])
    output_cb_depth = 2
    if "--output-cb-depth" in sys.argv:
        output_cb_depth = int(sys.argv[sys.argv.index("--output-cb-depth") + 1])
    workdir = Path(os.environ["TT_RQM_PERSISTENT_QMUL_DIR"])
    manifest = json.loads(
        Path(os.environ["TT_RQM_PERSISTENT_QMUL_MANIFEST"]).read_text()
    )
    cases = []
    for spec in manifest["cases"]:
        a = read_f32(workdir / spec["inputs"]["a"])
        b = read_f32(workdir / spec["inputs"]["b"])
        out = array("f")
        for offset in range(0, len(a), 4):
            ar, ai, aj, ak = a[offset : offset + 4]
            br, bi, bj, bk = b[offset : offset + 4]
            out.extend(
                (
                    ar * br - ai * bi - aj * bj - ak * bk,
                    ar * bi + ai * br + aj * bk - ak * bj,
                    ar * bj - ai * bk + aj * br + ak * bi,
                    ar * bk + ai * bj - aj * bi + ak * br,
                )
            )
        out_path = workdir / spec["outputs"]["out"]
        write_f32(out_path, out)
        tiles = (spec["items"] + 1023) // 1024
        requested = int(spec.get("requested_max_cores", min(tiles, 56)))
        active_cores = min(tiles, 56, requested)
        tiles_1 = (tiles + active_cores - 1) // active_cores
        group_1 = tiles % active_cores or active_cores
        group_2 = active_cores - group_1
        tiles_2 = tiles // active_cores if group_2 else 0
        timings = {
            "buffer_allocation": 0.000001,
            "program_build": 0.000001,
            "h2d": 0.000001,
            "prewarm_sync": 0.000001,
            "warmup": 0.000001,
            "samples": [0.000001] * spec["samples"],
            "d2h": 0.000001,
            "cleanup": 0.000001,
        }
        cases.append(
            {
                "case_id": spec["case_id"],
                "items": spec["items"],
                "iterations": spec["iterations"],
                "warmup": spec["warmup"],
                "samples": spec["samples"],
                "input_identity": {
                    "a_sha256": spec["inputs"]["a_sha256"],
                    "b_sha256": spec["inputs"]["b_sha256"],
                },
                "output_identity": {
                    "fnv1a64": fnv1a64(out_path.read_bytes()),
                    "value_count": len(out),
                },
                "timings_s": timings,
                "work": {
                    "device_count": 1,
                    "device_id": device_id,
                    "core_count": active_cores,
                    "requested_max_cores": requested,
                    "component_tiles": tiles,
                    "grid_x": 8,
                    "grid_y": 7,
                    "available_core_count": 56,
                    "group_1_core_count": group_1,
                    "group_2_core_count": group_2,
                    "group_1_tiles_per_core": tiles_1,
                    "group_2_tiles_per_core": tiles_2,
                    "work_allocation_imbalance_tiles": tiles_1 - tiles_2 if group_2 else 0,
                    "layout": "planar_float32_tiles_32x32",
                    "work_split": "row_major",
                    "arithmetic_path": "tensix_compute_sfpu",
                    "output_cb_depth": output_cb_depth,
                },
            }
        )
    provenance = {
        "chip_type": "wormhole-n300-fixture",
        "tt_metal_commit": "d" * 40,
        "compiler_version": "fixture-c++",
        "runtime_version": "fixture-metalium",
        "build_id": os.environ["TT_RQM_CANDIDATE_SHA256"],
        "candidate_sha256": os.environ["TT_RQM_CANDIDATE_SHA256"],
        "repository_commit": os.environ["TT_RQM_REPOSITORY_COMMIT"],
        "timer_scope": "fixture persistent device session",
    }
    metrics = {
        "schema": "tt-rqm-external-qmul-persistent-metrics.v1",
        "protocol": "tt-rqm-external-qmul-persistent.v1",
        "backend": "fixture",
        "device": f"tenstorrent/wormhole-device-{device_id}",
        "dtype": "float32",
        "execution_kind": "hardware",
        "implementation_class": "multicore_tensix_sfpu_qmul_persistent",
        "performance_eligible": True,
        "stable_benchmark": False,
        "lifecycle": {
            "device_count": 1,
            "device_id": device_id,
            "create_count": 1,
            "close_count": 1,
        },
        "session_timings_s": {
            "device_create": 0.000001,
            "device_close": 0.000001,
            "candidate_session": 0.001,
        },
        "cases": cases,
        "provenance": provenance,
    }
    if os.environ.get("TT_RQM_FIXTURE_BAD_DEVICE"):
        metrics["lifecycle"]["device_id"] = 1
    if os.environ.get("TT_RQM_FIXTURE_BAD_HASH"):
        metrics["provenance"]["candidate_sha256"] = "0" * 64
    (workdir / manifest["outputs"]["metrics"]).write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
