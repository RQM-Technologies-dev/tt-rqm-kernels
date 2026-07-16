#!/usr/bin/env python3
"""Validate a retained H2A hardware pilot entirely offline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tt_rqm_kernels.hamiltonian_lowering_pilot import (
    HamiltonianLoweringPilotError,
    validate_pilot_package,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    args = parser.parse_args()
    try:
        report = validate_pilot_package(args.package)
    except HamiltonianLoweringPilotError as exc:
        print(exc)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
