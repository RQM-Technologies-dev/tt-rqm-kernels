#!/usr/bin/env python3
"""Validate the frozen non-designated H2B pilot contract."""

from __future__ import annotations

import argparse
from pathlib import Path

from tt_rqm_kernels.hamiltonian_evolution_pilot_contract import (
    DEFAULT_MANIFEST,
    HamiltonianEvolutionPilotContractError,
    validate_pilot_contract,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / DEFAULT_MANIFEST)
    args = parser.parse_args()
    try:
        payload = validate_pilot_contract(args.manifest, ROOT)
    except HamiltonianEvolutionPilotContractError as exc:
        print(exc)
        return 1
    print(
        f"H2B pilot contract valid: status={payload['status']} "
        f"non_designated={str(payload['non_designated']).lower()} cases={len(payload['cases'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
