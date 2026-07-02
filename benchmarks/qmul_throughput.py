from __future__ import annotations

import argparse
import time

import torch

from tt_rqm_kernels import qmul, qnormalize


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark quaternion Hamilton product throughput.")
    parser.add_argument("--items", type=int, default=1_000_000)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    a = qnormalize(torch.randn(args.items, 4, device=args.device))
    b = qnormalize(torch.randn(args.items, 4, device=args.device))

    for _ in range(5):
        qmul(a, b)

    start = time.perf_counter()
    out = None
    for _ in range(args.iters):
        out = qmul(a, b)
    elapsed = time.perf_counter() - start

    products = args.items * args.iters
    print(f"items={args.items} iters={args.iters} device={args.device}")
    print(f"elapsed_s={elapsed:.6f}")
    print(f"qmul_per_s={products / elapsed:,.0f}")
    print(f"checksum={float(out.sum()):.6f}")


if __name__ == "__main__":
    main()
