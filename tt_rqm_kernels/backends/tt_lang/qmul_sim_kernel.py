"""TT-Lang simulator program for `[N, 4]` quaternion multiply.

Run this file through `tt-lang-sim`, not plain Python:

    tt-lang-sim tt_rqm_kernels/backends/tt_lang/qmul_sim_kernel.py --items 128
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

try:
    import ttl
    import ttnn
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by CLI use.
    raise SystemExit(
        "qmul_sim_kernel.py must be run with tt-lang-sim so simulator modules "
        "`ttl` and `ttnn` are injected."
    ) from exc

from tt_rqm_kernels.benchmark_integrity import validate_qmul_output, validate_report
from tt_rqm_kernels.quaternion_ops import qnormalize
from tt_rqm_kernels.structuredbench import (
    BenchmarkCase,
    BenchmarkResult,
    SCHEMA_VERSION,
    _error_metrics,
    _hardware_estimate,
    render_markdown_report,
)

BACKEND_NAME = "tt-lang-sim"
BLOCK_ITEMS = 32


@ttl.operation(grid=(1, 1))
def _qmul_operation(a: ttnn.Tensor, b: ttnn.Tensor, out: ttnn.Tensor) -> None:
    rows = a.shape[0] // BLOCK_ITEMS

    ar_dfb = ttl.make_dataflow_buffer_like(a, shape=(BLOCK_ITEMS, 1), block_count=2)
    ai_dfb = ttl.make_dataflow_buffer_like(a, shape=(BLOCK_ITEMS, 1), block_count=2)
    aj_dfb = ttl.make_dataflow_buffer_like(a, shape=(BLOCK_ITEMS, 1), block_count=2)
    ak_dfb = ttl.make_dataflow_buffer_like(a, shape=(BLOCK_ITEMS, 1), block_count=2)
    br_dfb = ttl.make_dataflow_buffer_like(b, shape=(BLOCK_ITEMS, 1), block_count=2)
    bi_dfb = ttl.make_dataflow_buffer_like(b, shape=(BLOCK_ITEMS, 1), block_count=2)
    bj_dfb = ttl.make_dataflow_buffer_like(b, shape=(BLOCK_ITEMS, 1), block_count=2)
    bk_dfb = ttl.make_dataflow_buffer_like(b, shape=(BLOCK_ITEMS, 1), block_count=2)
    or_dfb = ttl.make_dataflow_buffer_like(out, shape=(BLOCK_ITEMS, 1), block_count=2)
    oi_dfb = ttl.make_dataflow_buffer_like(out, shape=(BLOCK_ITEMS, 1), block_count=2)
    oj_dfb = ttl.make_dataflow_buffer_like(out, shape=(BLOCK_ITEMS, 1), block_count=2)
    ok_dfb = ttl.make_dataflow_buffer_like(out, shape=(BLOCK_ITEMS, 1), block_count=2)

    @ttl.datamovement()
    def read() -> None:
        for row in range(rows):
            start = row * BLOCK_ITEMS
            end = start + BLOCK_ITEMS
            with (
                ar_dfb.reserve() as ar_blk,
                ai_dfb.reserve() as ai_blk,
                aj_dfb.reserve() as aj_blk,
                ak_dfb.reserve() as ak_blk,
                br_dfb.reserve() as br_blk,
                bi_dfb.reserve() as bi_blk,
                bj_dfb.reserve() as bj_blk,
                bk_dfb.reserve() as bk_blk,
            ):
                tx_ar = ttl.copy(a[start:end, 0:1], ar_blk)
                tx_ai = ttl.copy(a[start:end, 1:2], ai_blk)
                tx_aj = ttl.copy(a[start:end, 2:3], aj_blk)
                tx_ak = ttl.copy(a[start:end, 3:4], ak_blk)
                tx_br = ttl.copy(b[start:end, 0:1], br_blk)
                tx_bi = ttl.copy(b[start:end, 1:2], bi_blk)
                tx_bj = ttl.copy(b[start:end, 2:3], bj_blk)
                tx_bk = ttl.copy(b[start:end, 3:4], bk_blk)
                tx_ar.wait()
                tx_ai.wait()
                tx_aj.wait()
                tx_ak.wait()
                tx_br.wait()
                tx_bi.wait()
                tx_bj.wait()
                tx_bk.wait()

    @ttl.compute()
    def compute() -> None:
        for _ in range(rows):
            with (
                ar_dfb.wait() as ar_blk,
                ai_dfb.wait() as ai_blk,
                aj_dfb.wait() as aj_blk,
                ak_dfb.wait() as ak_blk,
                br_dfb.wait() as br_blk,
                bi_dfb.wait() as bi_blk,
                bj_dfb.wait() as bj_blk,
                bk_dfb.wait() as bk_blk,
                or_dfb.reserve() as or_blk,
                oi_dfb.reserve() as oi_blk,
                oj_dfb.reserve() as oj_blk,
                ok_dfb.reserve() as ok_blk,
            ):
                or_blk.store(
                    ar_blk * br_blk - ai_blk * bi_blk - aj_blk * bj_blk - ak_blk * bk_blk
                )
                oi_blk.store(
                    ar_blk * bi_blk + ai_blk * br_blk + aj_blk * bk_blk - ak_blk * bj_blk
                )
                oj_blk.store(
                    ar_blk * bj_blk - ai_blk * bk_blk + aj_blk * br_blk + ak_blk * bi_blk
                )
                ok_blk.store(
                    ar_blk * bk_blk + ai_blk * bj_blk - aj_blk * bi_blk + ak_blk * br_blk
                )

    @ttl.datamovement()
    def write() -> None:
        for row in range(rows):
            start = row * BLOCK_ITEMS
            end = start + BLOCK_ITEMS
            with (
                or_dfb.wait() as or_blk,
                oi_dfb.wait() as oi_blk,
                oj_dfb.wait() as oj_blk,
                ok_dfb.wait() as ok_blk,
            ):
                tx_or = ttl.copy(or_blk, out[start:end, 0:1])
                tx_oi = ttl.copy(oi_blk, out[start:end, 1:2])
                tx_oj = ttl.copy(oj_blk, out[start:end, 2:3])
                tx_ok = ttl.copy(ok_blk, out[start:end, 3:4])
                tx_or.wait()
                tx_oi.wait()
                tx_oj.wait()
                tx_ok.wait()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a TT-Lang simulator qmul smoke benchmark."
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
            _qmul_operation(a_tt, b_tt, out_tt)

        start = time.perf_counter()
        for _ in range(iterations):
            _qmul_operation(a_tt, b_tt, out_tt)
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
            "variant": "block-slice",
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


def _from_torch(tensor: torch.Tensor, device: object) -> ttnn.Tensor:
    return ttnn.from_torch(
        tensor,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _result_from_output(
    *,
    case: BenchmarkCase,
    output: torch.Tensor,
    reference: torch.Tensor,
    elapsed_s: float,
    scalar_reference_max_abs_error: float,
    correctness: dict[str, object],
) -> BenchmarkResult:
    errors = _error_metrics(output, reference)
    hardware = _hardware_estimate(case, torch.float32, elapsed_s)
    return BenchmarkResult(
        suite="qmul",
        workload="qmul",
        backend=BACKEND_NAME,
        device="functional-simulator",
        dtype="float32",
        items=case.items,
        iterations=case.iterations,
        warmup=case.warmup,
        structured_shape=f"[{case.items}, 4]",
        throughput_unit=case.throughput_unit,
        elapsed_s=elapsed_s,
        latency_ms=elapsed_s * 1000.0 / case.iterations,
        throughput=case.items * case.iterations / elapsed_s,
        max_abs_error=errors["max_abs_error"],
        max_rel_error=errors["max_rel_error"],
        rms_error=errors["rms_error"],
        stability_max_abs=None,
        scalar_reference_max_abs_error=scalar_reference_max_abs_error,
        estimated_flops=hardware["estimated_flops"],
        estimated_flops_per_s=hardware["estimated_flops_per_s"],
        estimated_bytes_read=hardware["estimated_bytes_read"],
        estimated_bytes_written=hardware["estimated_bytes_written"],
        estimated_total_bytes=hardware["estimated_total_bytes"],
        effective_gb_per_s=hardware["effective_gb_per_s"],
        arithmetic_intensity_flops_per_byte=hardware[
            "arithmetic_intensity_flops_per_byte"
        ],
        checksum=float(output.detach().cpu().to(torch.float64).sum().item()),
        execution_label="simulator",
        stable_benchmark=False,
        methodology_note="TT-Lang functional simulator run; not hardware performance.",
        correctness=correctness,
        timing={
            "repetitions": 1,
            "device_s": {
                "samples": [elapsed_s],
                "median": elapsed_s,
                "p95": elapsed_s,
            },
            "primary_elapsed": "device_s.median",
        },
    )


def _pad_rows(values: torch.Tensor, target_rows: int) -> torch.Tensor:
    if values.shape[0] == target_rows:
        return values
    padding = torch.zeros(
        (target_rows - values.shape[0], values.shape[1]),
        dtype=values.dtype,
        device=values.device,
    )
    return torch.cat([values, padding], dim=0)


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
