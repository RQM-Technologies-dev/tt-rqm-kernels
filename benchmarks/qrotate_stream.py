from __future__ import annotations

import argparse
import time

import torch

from tt_rqm_kernels import qnormalize, qrotate_vector


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark streamed vector rotation by unit rotors.")
    parser.add_argument("--items", type=int, default=500_000)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    rotors = qnormalize(torch.randn(args.items, 4, device=args.device))
    vectors = torch.randn(args.items, 3, device=args.device)

    for _ in range(3):
        qrotate_vector(rotors, vectors, assume_unit=True)

    start = time.perf_counter()
    out = None
    for _ in range(args.iters):
        out = qrotate_vector(rotors, vectors, assume_unit=True)
    elapsed = time.perf_counter() - start

    rotations = args.items * args.iters
    print(f"items={args.items} iters={args.iters} device={args.device}")
    print(f"elapsed_s={elapsed:.6f}")
    print(f"rotations_per_s={rotations / elapsed:,.0f}")
    print(f"checksum={float(out.sum()):.6f}")


if __name__ == "__main__":
    main()
