#!/usr/bin/env python3
"""Direct qmul quickstart for CPU reference and optional Tenstorrent candidates.

The JSON/Markdown files from this example are lightweight quickstart reports.
For canonical StructuredBench reports, use scripts/rqm_tt_quickstart.py with
--mode emule or --mode hardware.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from tt_rqm_kernels.backends import torch_backend
from tt_rqm_kernels.backends.tenstorrent.availability import resolve_execution_path
from tt_rqm_kernels.backends.tenstorrent.qmul_external import (
    TenstorrentAdapterError,
    run_external_qmul_inputs,
)
from tt_rqm_kernels.backends.tenstorrent.report import (
    ReportLabelError,
    validate_external_qmul_label,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic [N, 4] quaternion tensors, run CPU/PyTorch "
            "qmul, and optionally compare an external Tenstorrent qmul command."
        )
    )
    parser.add_argument("--items", type=_positive_int, default=128)
    parser.add_argument("--iters", type=_positive_int, default=1)
    parser.add_argument("--warmup", type=_nonnegative_int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=("cpu", "emule", "hardware"), default="cpu")
    parser.add_argument("--command", default=None)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help=(
            "Write a lightweight quickstart JSON report. For canonical "
            "StructuredBench reports, use scripts/rqm_tt_quickstart.py."
        ),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=None,
        help=(
            "Write a lightweight quickstart Markdown report. For canonical "
            "StructuredBench reports, use scripts/rqm_tt_quickstart.py."
        ),
    )
    args = parser.parse_args()

    a, b = _inputs(args.items, args.seed)
    cpu = _run_cpu(a, b, iterations=args.iters, warmup=args.warmup)
    report = {
        "schema": "tt-rqm-tenstorrent-qmul-quickstart.v1",
        "items": args.items,
        "iterations": args.iters,
        "warmup": args.warmup,
        "seed": args.seed,
        "report_note": (
            "Lightweight quickstart report. Use scripts/rqm_tt_quickstart.py "
            "--mode emule or --mode hardware for canonical StructuredBench reports."
        ),
        "cpu": cpu,
        "tenstorrent": None,
    }

    print("Tenstorrent qmul quickstart")
    print(f"items: {args.items}")
    print(f"CPU/PyTorch latency_ms: {cpu['latency_ms']:.6f}")
    print(f"CPU/PyTorch throughput_qmul_s: {cpu['throughput']:.2f}")
    print(f"CPU/PyTorch checksum: {cpu['checksum']:.8f}")

    if args.mode != "cpu":
        path = resolve_execution_path(args.mode, command=args.command)
        if not path.available or not path.command:
            raise SystemExit(path.reason)
        try:
            validate_external_qmul_label(path.execution_label, command=path.command)
        except ReportLabelError as exc:
            raise SystemExit(str(exc)) from None
        try:
            external = run_external_qmul_inputs(
                a,
                b,
                command=path.command,
                iterations=args.iters,
                warmup=args.warmup,
                seed=args.seed,
            )
        except TenstorrentAdapterError as exc:
            raise SystemExit(str(exc)) from None
        external_payload = {
            "mode": args.mode,
            "execution_label": path.execution_label,
            "device": external.device,
            "latency_ms": external.latency_ms,
            "throughput": external.throughput,
            "max_abs_error": external.max_abs_error,
            "rms_error": external.rms_error,
            "checksum": external.checksum,
            "metrics": external.metrics,
        }
        report["tenstorrent"] = external_payload
        print(f"{args.mode} device: {external.device}")
        print(f"{args.mode} execution_label: {path.execution_label}")
        print(f"{args.mode} latency_ms: {external.latency_ms:.6f}")
        print(f"{args.mode} throughput_qmul_s: {external.throughput:.2f}")
        print(f"{args.mode} max_abs_error: {external.max_abs_error:.6e}")
        print(f"{args.mode} rms_error: {external.rms_error:.6e}")
        print(f"{args.mode} checksum: {external.checksum:.8f}")
    else:
        print("Tenstorrent external path: not run; pass --mode emule or --mode hardware.")

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(_render_markdown(report), encoding="utf-8")
    return 0


def _inputs(items: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    a = torch_backend.qnormalize(torch.randn((items, 4), generator=generator))
    b = torch_backend.qnormalize(torch.randn((items, 4), generator=generator))
    return a.to(dtype=torch.float32), b.to(dtype=torch.float32)


def _run_cpu(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    iterations: int,
    warmup: int,
) -> dict[str, float]:
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
    return {
        "latency_ms": (elapsed_s / iterations) * 1000.0,
        "throughput": (a.shape[0] * iterations) / elapsed_s,
        "checksum": float(output.to(dtype=torch.float64).sum().item()),
    }


def _render_markdown(report: dict[str, object]) -> str:
    lines = [
        "# Tenstorrent qmul Quickstart",
        "",
        f"items: `{report['items']}`  ",
        f"iterations: `{report['iterations']}`  ",
        f"warmup: `{report['warmup']}`  ",
        f"seed: `{report['seed']}`",
        "",
        (
            "This is a lightweight quickstart report. Use "
            "`scripts/rqm_tt_quickstart.py --mode emule` or "
            "`scripts/rqm_tt_quickstart.py --mode hardware` for canonical "
            "StructuredBench reports."
        ),
        "",
        "## CPU/PyTorch",
        "",
    ]
    cpu = report["cpu"]
    assert isinstance(cpu, dict)
    lines.extend(
        [
            f"- latency_ms: `{float(cpu['latency_ms']):.6f}`",
            f"- throughput_qmul_s: `{float(cpu['throughput']):.2f}`",
            f"- checksum: `{float(cpu['checksum']):.8f}`",
            "",
        ]
    )
    tenstorrent = report.get("tenstorrent")
    if isinstance(tenstorrent, dict):
        lines.extend(
            [
                "## Tenstorrent External qmul",
                "",
                f"- mode: `{tenstorrent['mode']}`",
                f"- execution_label: `{tenstorrent['execution_label']}`",
                f"- device: `{tenstorrent['device']}`",
                f"- latency_ms: `{float(tenstorrent['latency_ms']):.6f}`",
                f"- throughput_qmul_s: `{float(tenstorrent['throughput']):.2f}`",
                f"- max_abs_error: `{float(tenstorrent['max_abs_error']):.6e}`",
                f"- rms_error: `{float(tenstorrent['rms_error']):.6e}`",
                f"- checksum: `{float(tenstorrent['checksum']):.8f}`",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Tenstorrent External qmul",
                "",
                "Not run. Pass `--mode emule` or `--mode hardware` with a configured command.",
                "",
            ]
        )
    return "\n".join(lines)


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


if __name__ == "__main__":
    raise SystemExit(main())
