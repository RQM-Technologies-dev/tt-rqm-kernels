#!/usr/bin/env python3
"""Capture a sanitized, contract-bound H2B N300 preflight report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tt_rqm_kernels.hamiltonian_evolution_pilot import run_pilot_preflight
from tt_rqm_kernels.hamiltonian_evolution_pilot_contract import (
    DEFAULT_MANIFEST,
    validate_pilot_contract,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/h2b_n300_session2_preflight.json",
    )
    args = parser.parse_args()
    contract = validate_pilot_contract(ROOT / DEFAULT_MANIFEST, ROOT)
    payload = run_pilot_preflight(args.command, contract)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("H2B Session 2 preflight passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
