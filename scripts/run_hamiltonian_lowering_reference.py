#!/usr/bin/env python3
"""Run the deterministic H2A CPU-reference benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tt_rqm_kernels.hamiltonian_lowering_benchmark import run_reference_benchmark


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_reference_benchmark(seed=args.seed, iterations=args.iterations)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
