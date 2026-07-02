from __future__ import annotations

import argparse
import time

import torch

from tt_rqm_kernels import qmul, qnormalize


def qmatvec(matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    products = qmul(matrix, vector.unsqueeze(-3))
    return products.sum(dim=-2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark quaternion matrix-vector reference math.")
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--cols", type=int, default=256)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--iters", type=int, default=25)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    matrix = torch.randn(args.batch, args.rows, args.cols, 4, device=args.device)
    vector = qnormalize(torch.randn(args.batch, args.cols, 4, device=args.device))

    for _ in range(3):
        qmatvec(matrix, vector)

    start = time.perf_counter()
    out = None
    for _ in range(args.iters):
        out = qmatvec(matrix, vector)
    elapsed = time.perf_counter() - start

    products = args.batch * args.rows * args.cols * args.iters
    print(
        f"batch={args.batch} rows={args.rows} cols={args.cols} "
        f"iters={args.iters} device={args.device}"
    )
    print(f"elapsed_s={elapsed:.6f}")
    print(f"qmul_terms_per_s={products / elapsed:,.0f}")
    print(f"checksum={float(out.sum()):.6f}")


if __name__ == "__main__":
    main()
