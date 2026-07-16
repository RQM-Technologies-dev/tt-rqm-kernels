#!/usr/bin/env python3
"""Freeze the exact non-designated H2B pilot contract after reproducible build."""

from __future__ import annotations

import argparse
from pathlib import Path

from tt_rqm_kernels.hamiltonian_evolution_pilot_contract import (
    DEFAULT_MANIFEST,
    write_pilot_contract,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / DEFAULT_MANIFEST)
    args = parser.parse_args()
    payload = write_pilot_contract(args.output, ROOT, args.candidate_sha256)
    print(
        f"H2B pilot contract frozen: cases={len(payload['cases'])} "
        f"candidate={payload['candidate_binary_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
