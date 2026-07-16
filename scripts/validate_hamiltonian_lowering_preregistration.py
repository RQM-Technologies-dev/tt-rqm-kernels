#!/usr/bin/env python3
"""Validate the pre-hardware H2A Claim Level 0 methodology."""

from __future__ import annotations

import argparse
from pathlib import Path

from tt_rqm_kernels.hamiltonian_lowering_preregistration import load_preregistration


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("benchmarks/manifests/hamiltonian-lowering-h2a-preregistration.json"),
    )
    args = parser.parse_args()
    manifest = load_preregistration(args.manifest)
    print(
        "HamiltonianLoweringBench H2A preregistration valid "
        f"({len(manifest['cases'])} cases, status={manifest['status']}, target=Level 0)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
