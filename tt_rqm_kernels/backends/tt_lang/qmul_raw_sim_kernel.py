"""Experimental TT-Lang raw-element simulator program for quaternion multiply.

Run this file through `tt-lang-sim`, not plain Python:

    tt-lang-sim tt_rqm_kernels/backends/tt_lang/qmul_raw_sim_kernel.py --items 128

This is a simulator-only comparison variant. It uses TT-Lang raw element reads
and writes inside a data-movement kernel to make the scalar lane dependencies
explicit. It is not TT-Metalium source and it is not hardware performance
evidence.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from tt_rqm_kernels.benchmark_integrity import validate_qmul_output, validate_report

try:
    import ttl
    import ttnn
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by CLI use.
    raise SystemExit(
        "qmul_raw_sim_kernel.py must be run with tt-lang-sim so simulator "
        "modules `ttl` and `ttnn` are injected."
    ) from exc

from tt_rqm_kernels.backends.tt_lang.qmul_sim_kernel import (
    BACKEND_NAME,
    BLOCK_ITEMS,
    _from_torch,
    _nonnegative_int,
    _pad_rows,
    _positive_int,
    _result_from_output,
    _round_up,
    _write_text,
)
from tt_rqm_kernels.quaternion_ops import qnormalize
from tt_rqm_kernels.structuredbench import (
    BenchmarkCase,
    SCHEMA_VERSION,
    render_markdown_report,
)

VARIANT_NAME = "raw-element"


@ttl.operation(grid=(1, 1))
def _qmul_raw_operation(a: ttnn.Tensor, b: ttnn.Tensor, out: ttnn.Tensor) -> None:
    rows = a.shape[0] // BLOCK_ITEMS

    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(BLOCK_ITEMS, 4), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(BLOCK_ITEMS, 4), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(BLOCK_ITEMS, 4), block_count=2)

    @ttl.datamovement()
    def raw_qmul() -> None:
        for row in range(rows):
            start = row * BLOCK_ITEMS
            end = start + BLOCK_ITEMS

            with (
                a_dfb.reserve() as a_blk,
                b_dfb.reserve() as b_blk,
                out_dfb.reserve() as out_blk,
            ):
                tx_a = ttl.copy(a[start:end, 0:4], a_blk)
                tx_b = ttl.copy(b[start:end, 0:4], b_blk)
                tx_a.wait()
                tx_b.wait()

                for item in range(BLOCK_ITEMS):
                    ar = ttl.raw_element_read(a_blk, item, 0)
                    ai = ttl.raw_element_read(a_blk, item, 1)
                    aj = ttl.raw_element_read(a_blk, item, 2)
                    ak = ttl.raw_element_read(a_blk, item, 3)
                    br = ttl.raw_element_read(b_blk, item, 0)
                    bi = ttl.raw_element_read(b_blk, item, 1)
                    bj = ttl.raw_element_read(b_blk, item, 2)
                    bk = ttl.raw_element_read(b_blk, item, 3)

                    real = ar * br - ai * bi - aj * bj - ak * bk
                    i_lane = ar * bi + ai * br + aj * bk - ak * bj
                    j_lane = ar * bj - ai * bk + aj * br + ak * bi
                    k_lane = ar * bk + ai * bj - aj * bi + ak * br

                    ttl.raw_element_write(out_blk, item, 0, value=real)
                    ttl.raw_element_write(out_blk, item, 1, value=i_lane)
                    ttl.raw_element_write(out_blk, item, 2, value=j_lane)
                    ttl.raw_element_write(out_blk, item, 3, value=k_lane)

                tx_out = ttl.copy(out_blk, out[start:end, 0:4])
                tx_out.wait()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run an experimental TT-Lang raw-element qmul simulator benchmark."
    )
    parser.add_argument("--items", type=_positive_int, default=128)
    parser.add_argument("--iters", type=_positive_int, default=1)
    parser.add_argument("--warmup", type=_nonnegative_int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--markdown-output", type=Path, default=None)
    args = parser.parse_args(argv)

    report = run_qmul_report(
        items=args.items,
        iterations=args.iters,
        warmup=args.warmup,
        seed=args.seed,
    )

    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.json_output is not None:
        _write_text(args.json_output, rendered + "\n")
    if args.markdown_output is not None:
        _write_text(args.markdown_output, render_markdown_report(report))

    print(rendered)
    return 0


def run_qmul_report(
    *,
    items: int,
    iterations: int,
    warmup: int,
    seed: int,
) -> dict[str, object]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    a64 = qnormalize(torch.randn((items, 4), generator=generator, dtype=torch.float64))
    b64 = qnormalize(torch.randn((items, 4), generator=generator, dtype=torch.float64))
    padded_items = _round_up(items, BLOCK_ITEMS)
    a_padded = _pad_rows(a64.to(torch.float32), padded_items)
    b_padded = _pad_rows(b64.to(torch.float32), padded_items)

    device = ttnn.open_device(device_id=0)
    try:
        a_tt = _from_torch(a_padded, device)
        b_tt = _from_torch(b_padded, device)
        out_tt = _from_torch(torch.zeros((padded_items, 4), dtype=torch.float32), device)

        for _ in range(warmup):
            _qmul_raw_operation(a_tt, b_tt, out_tt)

        start = time.perf_counter()
        for _ in range(iterations):
            _qmul_raw_operation(a_tt, b_tt, out_tt)
        elapsed_s = time.perf_counter() - start

        output = ttnn.to_torch(out_tt)[:items].to(torch.float32)
    finally:
        ttnn.close_device(device)

    reference, correctness = validate_qmul_output(
        output, a_padded[:items], b_padded[:items]
    )
    scalar_error = float(correctness["scalar_first_eight_max_abs_error"])
    case = BenchmarkCase(
        workload="qmul",
        items=items,
        iterations=iterations,
        warmup=warmup,
        throughput_unit="qmul/s",
    )
    result = _result_from_output(
        case=case,
        output=output,
        reference=reference,
        elapsed_s=elapsed_s,
        scalar_reference_max_abs_error=scalar_error,
        correctness=correctness,
    )
    report = {
        "schema": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "suite": "qmul",
        "backend": BACKEND_NAME,
        "device": "functional-simulator",
        "execution_label": "simulator",
        "stable_benchmark": False,
        "methodology_note": "TT-Lang functional simulator run; not hardware performance.",
        "dtype": "float32",
        "seed": seed,
        "simulation": True,
        "tt_lang_sim": {
            "block_items": BLOCK_ITEMS,
            "padded_items": padded_items,
            "layout": "row-major",
            "variant": VARIANT_NAME,
            "variant_note": (
                "experimental raw-element datamovement variant; simulator-only"
            ),
        },
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "results": [asdict(result)],
        "repetitions": 1,
        "case_items": [items],
    }
    validate_report(report)
    return report


if __name__ == "__main__":
    raise SystemExit(main())
