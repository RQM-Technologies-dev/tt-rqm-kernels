#!/usr/bin/env python3
"""Collect one disclosed non-designated H2A hardware pilot."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tt_rqm_kernels.hamiltonian_lowering_pilot import collect_pilot


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pilot-id", required=True)
    args = parser.parse_args()
    report = collect_pilot(command=args.command, output_dir=args.output_dir, pilot_id=args.pilot_id)
    print(f"pilot_id={report['pilot_id']}")
    print(f"suite_passed={str(report['suite_passed']).lower()}")
    return 0 if report["suite_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
