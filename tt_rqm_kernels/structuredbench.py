"""StructuredBench benchmark CLI for structured quaternion tensor kernels."""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch

from tt_rqm_kernels.phase_ops import integrate_phase, phase_to_unit_vector
from tt_rqm_kernels.quaternion_ops import qinverse, qmul, qnorm, qnormalize
from tt_rqm_kernels.rotor_ops import qrotate_vector

SCHEMA_VERSION = "structuredbench.v1"
SUPPORTED_SUITES = ("smoke", "full", "qmul", "qrotate")
SUPPORTED_DTYPES = {
    "float32": torch.float32,
    "float64": torch.float64,
}


@dataclass(frozen=True)
class BenchmarkCase:
    """One benchmark configuration."""

    workload: str
    items: int
    iterations: int
    warmup: int
    throughput_unit: str


@dataclass(frozen=True)
class BenchmarkResult:
    """One measured benchmark result."""

    suite: str
    workload: str
    backend: str
    device: str
    dtype: str
    items: int
    iterations: int
    warmup: int
    structured_shape: str
    throughput_unit: str
    elapsed_s: float
    latency_ms: float
    throughput: float
    max_abs_error: float
    max_rel_error: float
    rms_error: float
    stability_max_abs: float | None
    checksum: float


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def build_cases(
    suite: str,
    *,
    items_override: int | None = None,
    iterations_override: int | None = None,
    warmup_override: int | None = None,
) -> list[BenchmarkCase]:
    """Build benchmark cases for a named suite."""

    if suite == "smoke":
        cases = [
            BenchmarkCase("qmul", 1024, 5, 2, "qmul/s"),
            BenchmarkCase("qrotate", 1024, 5, 2, "rotations/s"),
            BenchmarkCase("qnormalize", 1024, 5, 2, "normalizations/s"),
            BenchmarkCase("qinverse", 1024, 5, 2, "inverses/s"),
            BenchmarkCase("phase_update", 2048, 5, 2, "phase-updates/s"),
        ]
    elif suite == "qmul":
        cases = [
            BenchmarkCase("qmul", 4096, 30, 5, "qmul/s"),
            BenchmarkCase("qmul", 65536, 30, 5, "qmul/s"),
            BenchmarkCase("qmul", 262144, 30, 5, "qmul/s"),
        ]
    elif suite == "qrotate":
        cases = [
            BenchmarkCase("qrotate", 4096, 20, 5, "rotations/s"),
            BenchmarkCase("qrotate", 65536, 20, 5, "rotations/s"),
            BenchmarkCase("qrotate", 262144, 20, 5, "rotations/s"),
        ]
    elif suite == "full":
        cases = [
            BenchmarkCase("qmul", 4096, 30, 5, "qmul/s"),
            BenchmarkCase("qmul", 65536, 30, 5, "qmul/s"),
            BenchmarkCase("qmul", 262144, 30, 5, "qmul/s"),
            BenchmarkCase("qrotate", 4096, 20, 5, "rotations/s"),
            BenchmarkCase("qrotate", 65536, 20, 5, "rotations/s"),
            BenchmarkCase("qrotate", 262144, 20, 5, "rotations/s"),
            BenchmarkCase("qnormalize", 4096, 30, 5, "normalizations/s"),
            BenchmarkCase("qnormalize", 65536, 30, 5, "normalizations/s"),
            BenchmarkCase("qnormalize", 262144, 30, 5, "normalizations/s"),
            BenchmarkCase("qinverse", 4096, 20, 5, "inverses/s"),
            BenchmarkCase("qinverse", 65536, 20, 5, "inverses/s"),
            BenchmarkCase("qinverse", 262144, 20, 5, "inverses/s"),
            BenchmarkCase("phase_update", 8192, 30, 5, "phase-updates/s"),
            BenchmarkCase("phase_update", 131072, 30, 5, "phase-updates/s"),
            BenchmarkCase("phase_update", 524288, 30, 5, "phase-updates/s"),
        ]
    else:
        raise ValueError(f"unsupported suite: {suite}")

    return [
        BenchmarkCase(
            workload=case.workload,
            items=items_override or case.items,
            iterations=iterations_override or case.iterations,
            warmup=case.warmup if warmup_override is None else warmup_override,
            throughput_unit=case.throughput_unit,
        )
        for case in cases
    ]


def run_suite(
    suite: str,
    *,
    backend: str = "torch",
    device_name: str = "cpu",
    dtype_name: str = "float32",
    seed: int = 0,
    items_override: int | None = None,
    iterations_override: int | None = None,
    warmup_override: int | None = None,
) -> dict[str, object]:
    """Run a StructuredBench suite and return a JSON-serializable report."""

    if backend != "torch":
        raise ValueError("only the torch backend is implemented")
    if dtype_name not in SUPPORTED_DTYPES:
        raise ValueError(f"unsupported dtype: {dtype_name}")

    device = _resolve_device(device_name)
    dtype = SUPPORTED_DTYPES[dtype_name]
    cases = build_cases(
        suite,
        items_override=items_override,
        iterations_override=iterations_override,
        warmup_override=warmup_override,
    )

    results = [
        _run_case(
            suite=suite,
            case=case,
            backend=backend,
            device=device,
            dtype=dtype,
            dtype_name=dtype_name,
            seed=seed + index,
        )
        for index, case in enumerate(cases)
    ]

    return {
        "schema": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "suite": suite,
        "backend": backend,
        "device": str(device),
        "dtype": dtype_name,
        "seed": seed,
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "results": [asdict(result) for result in results],
    }


def render_table(report: dict[str, object]) -> str:
    """Render a report as a compact plain-text table."""

    results = report["results"]
    if not isinstance(results, list):
        raise TypeError("report results must be a list")

    lines = [
        "StructuredBench",
        (
            f"schema={report['schema']} suite={report['suite']} "
            f"backend={report['backend']} device={report['device']} dtype={report['dtype']}"
        ),
        "",
    ]
    headers = [
        "workload",
        "items",
        "iters",
        "latency_ms",
        "throughput",
        "unit",
        "max_abs_err",
        "rms_err",
        "stability",
    ]
    rows = []
    for result in results:
        if not isinstance(result, dict):
            raise TypeError("each result must be a dict")
        stability = result["stability_max_abs"]
        rows.append(
            [
                str(result["workload"]),
                str(result["items"]),
                str(result["iterations"]),
                f"{float(result['latency_ms']):.4f}",
                f"{float(result['throughput']):.2f}",
                str(result["throughput_unit"]),
                f"{float(result['max_abs_error']):.3e}",
                f"{float(result['rms_error']):.3e}",
                "-" if stability is None else f"{float(stability):.3e}",
            ]
        )

    widths = [
        max(len(headers[column]), *(len(row[column]) for row in rows))
        for column in range(len(headers))
    ]
    lines.append(_format_row(headers, widths))
    lines.append(_format_row(["-" * width for width in widths], widths))
    lines.extend(_format_row(row, widths) for row in rows)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run StructuredBench quaternion, rotor, and phase tensor benchmarks."
    )
    parser.add_argument("--suite", choices=SUPPORTED_SUITES, default="smoke")
    parser.add_argument("--backend", choices=("torch",), default="torch")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=tuple(SUPPORTED_DTYPES), default="float32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--items", type=positive_int, default=None)
    parser.add_argument("--iters", type=positive_int, default=None)
    parser.add_argument("--warmup", type=nonnegative_int, default=None)
    parser.add_argument("--threads", type=positive_int, default=None)
    parser.add_argument("--format", choices=("table", "json"), default="table")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.threads is not None:
        torch.set_num_threads(args.threads)

    report = run_suite(
        args.suite,
        backend=args.backend,
        device_name=args.device,
        dtype_name=args.dtype,
        seed=args.seed,
        items_override=args.items,
        iterations_override=args.iters,
        warmup_override=args.warmup,
    )
    rendered = (
        json.dumps(report, indent=2, sort_keys=True)
        if args.format == "json"
        else render_table(report)
    )

    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")

    print(rendered)
    return 0


def _resolve_device(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but torch.cuda.is_available() is false")
    return device


def _run_case(
    *,
    suite: str,
    case: BenchmarkCase,
    backend: str,
    device: torch.device,
    dtype: torch.dtype,
    dtype_name: str,
    seed: int,
) -> BenchmarkResult:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if case.workload == "qmul":
        return _run_qmul(suite, case, backend, device, dtype, dtype_name, generator)
    if case.workload == "qrotate":
        return _run_qrotate(suite, case, backend, device, dtype, dtype_name, generator)
    if case.workload == "qnormalize":
        return _run_qnormalize(suite, case, backend, device, dtype, dtype_name, generator)
    if case.workload == "qinverse":
        return _run_qinverse(suite, case, backend, device, dtype, dtype_name, generator)
    if case.workload == "phase_update":
        return _run_phase_update(suite, case, backend, device, dtype, dtype_name, generator)
    raise ValueError(f"unsupported workload: {case.workload}")


def _run_qmul(
    suite: str,
    case: BenchmarkCase,
    backend: str,
    device: torch.device,
    dtype: torch.dtype,
    dtype_name: str,
    generator: torch.Generator,
) -> BenchmarkResult:
    a64 = qnormalize(_randn((case.items, 4), generator))
    b64 = qnormalize(_randn((case.items, 4), generator))
    a = _to_device_dtype(a64, device, dtype)
    b = _to_device_dtype(b64, device, dtype)

    output, elapsed_s = _measure(lambda: qmul(a, b), case.warmup, case.iterations, device)
    reference = qmul(a64, b64)
    return _result_from_output(
        suite,
        case,
        backend,
        device,
        dtype_name,
        output,
        reference,
        elapsed_s,
    )


def _run_qrotate(
    suite: str,
    case: BenchmarkCase,
    backend: str,
    device: torch.device,
    dtype: torch.dtype,
    dtype_name: str,
    generator: torch.Generator,
) -> BenchmarkResult:
    rotors64 = qnormalize(_randn((case.items, 4), generator))
    vectors64 = _randn((case.items, 3), generator)
    rotors = _to_device_dtype(rotors64, device, dtype)
    vectors = _to_device_dtype(vectors64, device, dtype)

    output, elapsed_s = _measure(
        lambda: qrotate_vector(rotors, vectors, assume_unit=True),
        case.warmup,
        case.iterations,
        device,
    )
    reference = qrotate_vector(rotors64, vectors64, assume_unit=True)
    stability = _max_abs(
        torch.linalg.vector_norm(output.detach().cpu().to(torch.float64), dim=-1)
        - torch.linalg.vector_norm(vectors64, dim=-1)
    )
    return _result_from_output(
        suite,
        case,
        backend,
        device,
        dtype_name,
        output,
        reference,
        elapsed_s,
        stability_max_abs=stability,
    )


def _run_qnormalize(
    suite: str,
    case: BenchmarkCase,
    backend: str,
    device: torch.device,
    dtype: torch.dtype,
    dtype_name: str,
    generator: torch.Generator,
) -> BenchmarkResult:
    values64 = _scaled_quaternions(case.items, generator, min_scale=1e-3, max_scale=1e3)
    values = _to_device_dtype(values64, device, dtype)

    output, elapsed_s = _measure(lambda: qnormalize(values), case.warmup, case.iterations, device)
    reference = qnormalize(values64)
    stability = _max_abs(qnorm(output.detach().cpu().to(torch.float64)) - 1.0)
    return _result_from_output(
        suite,
        case,
        backend,
        device,
        dtype_name,
        output,
        reference,
        elapsed_s,
        stability_max_abs=stability,
    )


def _run_qinverse(
    suite: str,
    case: BenchmarkCase,
    backend: str,
    device: torch.device,
    dtype: torch.dtype,
    dtype_name: str,
    generator: torch.Generator,
) -> BenchmarkResult:
    values64 = _scaled_quaternions(case.items, generator, min_scale=1e-1, max_scale=1e1)
    values = _to_device_dtype(values64, device, dtype)

    output, elapsed_s = _measure(lambda: qinverse(values), case.warmup, case.iterations, device)
    reference = qinverse(values64)
    residual = qmul(values, output).detach().cpu().to(torch.float64)
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64).expand_as(residual)
    stability = _max_abs(residual - identity)
    return _result_from_output(
        suite,
        case,
        backend,
        device,
        dtype_name,
        output,
        reference,
        elapsed_s,
        stability_max_abs=stability,
    )


def _run_phase_update(
    suite: str,
    case: BenchmarkCase,
    backend: str,
    device: torch.device,
    dtype: torch.dtype,
    dtype_name: str,
    generator: torch.Generator,
) -> BenchmarkResult:
    phase64 = _rand((case.items,), generator) * (2.0 * math.pi) - math.pi
    rate64 = _randn((case.items,), generator)
    amplitude64 = _rand((case.items,), generator)
    phase = _to_device_dtype(phase64, device, dtype)
    rate = _to_device_dtype(rate64, device, dtype)
    amplitude = _to_device_dtype(amplitude64, device, dtype)
    dt = 0.01

    def op() -> torch.Tensor:
        next_phase = integrate_phase(phase, rate, dt)
        return amplitude.unsqueeze(-1) * phase_to_unit_vector(next_phase)

    output, elapsed_s = _measure(op, case.warmup, case.iterations, device)
    reference_phase = integrate_phase(phase64, rate64, dt)
    reference = amplitude64.unsqueeze(-1) * phase_to_unit_vector(reference_phase)
    return _result_from_output(
        suite,
        case,
        backend,
        device,
        dtype_name,
        output,
        reference,
        elapsed_s,
    )


def _result_from_output(
    suite: str,
    case: BenchmarkCase,
    backend: str,
    device: torch.device,
    dtype_name: str,
    output: torch.Tensor,
    reference: torch.Tensor,
    elapsed_s: float,
    *,
    stability_max_abs: float | None = None,
) -> BenchmarkResult:
    errors = _error_metrics(output, reference)
    latency_ms = elapsed_s * 1000.0 / case.iterations
    throughput = case.items * case.iterations / elapsed_s
    checksum = float(output.detach().cpu().to(torch.float64).sum().item())
    return BenchmarkResult(
        suite=suite,
        workload=case.workload,
        backend=backend,
        device=str(device),
        dtype=dtype_name,
        items=case.items,
        iterations=case.iterations,
        warmup=case.warmup,
        structured_shape=_structured_shape(case),
        throughput_unit=case.throughput_unit,
        elapsed_s=elapsed_s,
        latency_ms=latency_ms,
        throughput=throughput,
        max_abs_error=errors["max_abs_error"],
        max_rel_error=errors["max_rel_error"],
        rms_error=errors["rms_error"],
        stability_max_abs=stability_max_abs,
        checksum=checksum,
    )


def _measure(
    op: Callable[[], torch.Tensor],
    warmup: int,
    iterations: int,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    output: torch.Tensor | None = None
    with torch.no_grad():
        for _ in range(warmup):
            output = op()
        _sync(device)
        start = time.perf_counter()
        for _ in range(iterations):
            output = op()
        _sync(device)
        elapsed_s = time.perf_counter() - start

    if output is None:
        output = op()
    return output, elapsed_s


def _error_metrics(output: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    output64 = output.detach().cpu().to(torch.float64)
    reference64 = reference.detach().cpu().to(torch.float64)
    diff = output64 - reference64
    abs_diff = torch.abs(diff)
    denom = torch.clamp(torch.abs(reference64), min=1e-12)
    rel_diff = abs_diff / denom
    return {
        "max_abs_error": _max_abs(diff),
        "max_rel_error": float(torch.max(rel_diff).item()),
        "rms_error": float(torch.sqrt(torch.mean(diff * diff)).item()),
    }


def _randn(shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    return torch.randn(shape, generator=generator, dtype=torch.float64)


def _rand(shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    return torch.rand(shape, generator=generator, dtype=torch.float64)


def _scaled_quaternions(
    items: int,
    generator: torch.Generator,
    *,
    min_scale: float,
    max_scale: float,
) -> torch.Tensor:
    values = _randn((items, 4), generator)
    scales = torch.logspace(
        math.log10(min_scale),
        math.log10(max_scale),
        steps=items,
        dtype=torch.float64,
    ).unsqueeze(-1)
    return values * scales


def _to_device_dtype(
    value: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return value.to(device=device, dtype=dtype)


def _max_abs(value: torch.Tensor) -> float:
    return float(torch.max(torch.abs(value)).item())


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _structured_shape(case: BenchmarkCase) -> str:
    if case.workload in {"qmul", "qnormalize", "qinverse"}:
        return f"[{case.items}, 4]"
    if case.workload == "qrotate":
        return f"rotor=[{case.items}, 4], vector=[{case.items}, 3]"
    if case.workload == "phase_update":
        return f"phase=[{case.items}], state=[{case.items}, 2]"
    return f"[{case.items}]"


def _format_row(values: list[str], widths: list[int]) -> str:
    return "  ".join(value.ljust(width) for value, width in zip(values, widths, strict=True))


if __name__ == "__main__":
    raise SystemExit(main())
