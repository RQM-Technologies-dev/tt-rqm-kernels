from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from tt_rqm_kernels import qmul, qnorm, qnormalize, qrotate_vector

SCHEMA = "tt-rqm-pose-stream-demo.v1"


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


def rotor_from_axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    axis = axis / torch.linalg.vector_norm(axis, dim=-1, keepdim=True).clamp_min(1e-12)
    half = angle.unsqueeze(-1) * 0.5
    return qnormalize(torch.cat((torch.cos(half), axis * torch.sin(half)), dim=-1))


def build_inputs(items: int, seed: int, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)

    base_axes = torch.randn((items, 3), generator=generator, dtype=torch.float64)
    delta_axes = torch.randn((items, 3), generator=generator, dtype=torch.float64)
    base_angles = (
        torch.rand((items,), generator=generator, dtype=torch.float64) * (2.0 * math.pi)
        - math.pi
    )
    delta_angles = torch.randn((items,), generator=generator, dtype=torch.float64) * 0.02
    body_vectors = torch.randn((items, 3), generator=generator, dtype=torch.float64)

    base_rotors = rotor_from_axis_angle(base_axes, base_angles).to(dtype=dtype)
    delta_rotors = rotor_from_axis_angle(delta_axes, delta_angles).to(dtype=dtype)
    body_vectors = body_vectors.to(dtype=dtype)

    return {
        "base_rotors": base_rotors,
        "delta_rotors": delta_rotors,
        "body_vectors": body_vectors,
    }


def run_pose_stream(
    *,
    items: int,
    iterations: int,
    warmup: int,
    seed: int,
    dtype: torch.dtype,
    dtype_name: str,
) -> dict[str, Any]:
    inputs = build_inputs(items, seed, dtype)
    base_rotors = inputs["base_rotors"]
    delta_rotors = inputs["delta_rotors"]
    body_vectors = inputs["body_vectors"]

    def step() -> tuple[torch.Tensor, torch.Tensor]:
        orientation = qnormalize(qmul(delta_rotors, base_rotors))
        world_vectors = qrotate_vector(orientation, body_vectors, assume_unit=True)
        return orientation, world_vectors

    for _ in range(warmup):
        step()

    start = time.perf_counter()
    orientation = world_vectors = None
    for _ in range(iterations):
        orientation, world_vectors = step()
    elapsed_s = time.perf_counter() - start

    if orientation is None or world_vectors is None:
        raise RuntimeError("pose stream produced no output")

    orientation64 = orientation.detach().to(torch.float64)
    world64 = world_vectors.detach().to(torch.float64)
    body64 = body_vectors.detach().to(torch.float64)

    unit_rotor_max_abs_error = float(torch.max(torch.abs(qnorm(orientation64) - 1.0)).item())
    norm_preservation_max_abs = float(
        torch.max(
            torch.abs(
                torch.linalg.vector_norm(world64, dim=-1)
                - torch.linalg.vector_norm(body64, dim=-1)
            )
        ).item()
    )

    latency_ms = elapsed_s * 1000.0 / iterations
    throughput = items * iterations / elapsed_s

    return {
        "schema": SCHEMA,
        "backend": "torch",
        "device": "cpu",
        "dtype": dtype_name,
        "seed": seed,
        "items": items,
        "iterations": iterations,
        "warmup": warmup,
        "structured_shape": (
            f"orientation=[{items}, 4], body_vector=[{items}, 3], "
            f"world_vector=[{items}, 3]"
        ),
        "throughput_unit": "pose-updates/s",
        "elapsed_s": elapsed_s,
        "latency_ms": latency_ms,
        "throughput": throughput,
        "unit_rotor_max_abs_error": unit_rotor_max_abs_error,
        "norm_preservation_max_abs": norm_preservation_max_abs,
        "checksum": float(world64.sum().item()),
        "notes": [
            "CPU/PyTorch reference demo only.",
            "This is not Tenstorrent hardware performance.",
            "Uses existing qmul, qnormalize, and qrotate_vector kernels.",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Physical-AI Pose Stream Demo",
            "",
            "CPU/PyTorch reference output only. This is not Tenstorrent hardware performance.",
            "",
            "| field | value |",
            "|---|---:|",
            f"| backend | {report['backend']} |",
            f"| device | {report['device']} |",
            f"| dtype | {report['dtype']} |",
            f"| items | {report['items']} |",
            f"| iterations | {report['iterations']} |",
            f"| latency_ms | {report['latency_ms']:.6f} |",
            f"| throughput | {report['throughput']:.2f} {report['throughput_unit']} |",
            f"| unit_rotor_max_abs_error | {report['unit_rotor_max_abs_error']:.3e} |",
            f"| norm_preservation_max_abs | {report['norm_preservation_max_abs']:.3e} |",
            f"| checksum | {report['checksum']:.12g} |",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a CPU/PyTorch physical-AI pose stream reference demo."
    )
    parser.add_argument("--items", type=positive_int, default=1024)
    parser.add_argument("--iters", type=positive_int, default=5)
    parser.add_argument("--warmup", type=nonnegative_int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()

    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    report = run_pose_stream(
        items=args.items,
        iterations=args.iters,
        warmup=args.warmup,
        seed=args.seed,
        dtype=dtype,
        dtype_name=args.dtype,
    )

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(report), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
