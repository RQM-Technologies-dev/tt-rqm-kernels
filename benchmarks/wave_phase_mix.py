from __future__ import annotations

import argparse
import time

import torch

from tt_rqm_kernels import integrate_phase, phase_to_unit_vector


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark simple phase-state wave mixing.")
    parser.add_argument("--items", type=int, default=1_000_000)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    amplitude = torch.rand(args.items, device=args.device)
    phase = torch.rand(args.items, device=args.device) * 2.0 * torch.pi - torch.pi
    angular_rate = torch.randn(args.items, device=args.device)

    start = time.perf_counter()
    out = None
    for _ in range(args.iters):
        phase = integrate_phase(phase, angular_rate, args.dt)
        out = amplitude.unsqueeze(-1) * phase_to_unit_vector(phase)
    elapsed = time.perf_counter() - start

    updates = args.items * args.iters
    print(f"items={args.items} iters={args.iters} device={args.device}")
    print(f"elapsed_s={elapsed:.6f}")
    print(f"phase_updates_per_s={updates / elapsed:,.0f}")
    print(f"checksum={float(out.sum()):.6f}")


if __name__ == "__main__":
    main()
