#!/usr/bin/env python3
"""Audit H2A-required compute APIs at the exact pinned TT-Metal commit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

PINNED_COMMIT = "dd2849b5bc6b7a5d38a9eafbeba31ef8d530f8d4"
CHECKS = {
    "tile_arithmetic": (
        "tt_metal/hw/inc/api/compute/eltwise_binary.h",
        ("add_tiles(", "mul_tiles("),
        "verified available",
    ),
    "sqrt": (
        "tt_metal/hw/inc/api/compute/eltwise_unary/sqrt.h",
        ("sqrt_tile_init", "sqrt_tile("),
        "available with restrictions",
    ),
    "reciprocal": (
        "tt_metal/hw/inc/api/compute/eltwise_unary/recip.h",
        ("recip_tile_init", "recip_tile("),
        "available with restrictions",
    ),
    "reciprocal_sqrt": (
        "tt_metal/hw/inc/api/compute/eltwise_unary/rsqrt.h",
        ("rsqrt_tile_init", "rsqrt_tile("),
        "available with restrictions",
    ),
    "trigonometry": (
        "tt_metal/hw/inc/api/compute/eltwise_unary/trigonometry.h",
        ("sin_tile_init", "sin_tile(", "cos_tile_init", "cos_tile("),
        "available with restrictions",
    ),
    "zero_comparison": (
        "tt_metal/hw/inc/api/compute/eltwise_unary/comp.h",
        ("eqz_tile_init", "eqz_tile("),
        "available with restrictions",
    ),
    "select": (
        "tt_metal/hw/inc/api/compute/eltwise_unary/where.h",
        ("where_tile_init", "where_tile(", "DataFormat::Int32", "DataFormat::UInt32"),
        "integer formats only; not selected for FP32 H2A",
    ),
    "negation": (
        "tt_metal/hw/inc/api/compute/eltwise_unary/negative.h",
        ("negative_tile_init", "negative_tile("),
        "verified available",
    ),
    "tile_pack": (
        "tt_metal/hw/inc/api/compute/pack.h",
        ("pack_tile(",),
        "verified available with acquired/committed DST lifecycle",
    ),
    "reader_writer_dma": (
        "tt_metal/hw/inc/api/dataflow/dataflow_api.h",
        ("noc_async_read_page", "noc_async_write_page", "noc_async_read_barrier"),
        "verified data-movement API",
    ),
    "circular_buffers": (
        "tt_metal/hw/inc/api/dataflow/circular_buffer.h",
        ("cb_wait_front", "cb_reserve_back", "cb_push_back", "cb_pop_front"),
        "verified producer/consumer API",
    ),
}


def audit(root: Path) -> dict[str, object]:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip()
    if commit != PINNED_COMMIT:
        raise ValueError(f"TT-Metal commit mismatch: expected {PINNED_COMMIT}, got {commit}")
    results: dict[str, object] = {}
    for name, (relative, symbols, classification) in CHECKS.items():
        path = root / relative
        if not path.is_file():
            raise ValueError(f"missing pinned API header: {relative}")
        text = path.read_text(encoding="utf-8")
        missing = [symbol for symbol in symbols if symbol not in text]
        if missing:
            raise ValueError(f"missing {name} symbols in {relative}: {missing}")
        results[name] = {
            "classification": classification,
            "path": relative,
            "symbols": list(symbols),
        }
    results["fp32_selection"] = {
        "classification": "custom SFPI lane selection required",
        "reason": "pinned where_tile documentation lists integer formats only",
    }
    results["vector_magnitude"] = {
        "classification": "composed FP32 square/add/sqrt implementation required"
    }
    results["native_trigonometry_range_reduction"] = {
        "classification": "four-stage Cody-Waite reduction present in pinned Wormhole source"
    }
    results["large_angle_accuracy"] = {
        "classification": "must still be verified end-to-end; range reduction presence is not a tolerance result"
    }
    return {
        "schema": "tt-rqm-h2a-tt-metal-api-audit.v1",
        "tt_metal_commit": commit,
        "checks": results,
        "hardware_execution_claim": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tt-metal-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = audit(args.tt_metal_root.resolve())
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        print(str(exc))
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
