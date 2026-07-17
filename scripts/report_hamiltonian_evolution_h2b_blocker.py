#!/usr/bin/env python3
"""Build the evidence-backed blocker report for a failed H2B pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tt_rqm_kernels.hamiltonian_evolution_pilot import (
    build_blocker_report,
    render_blocker_report,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/h2b_n300_pilot_session_2_blocker.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "reports/h2b_n300_pilot_session_2_blocker.md",
    )
    args = parser.parse_args()
    payload = build_blocker_report(args.package, ROOT)
    expected_json = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    expected_report = render_blocker_report(payload)
    if args.check:
        if args.output.read_text(encoding="utf-8") != expected_json:
            return 1
        if args.report.read_text(encoding="utf-8") != expected_report:
            return 1
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(expected_json, encoding="utf-8")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(expected_report, encoding="utf-8")
    print("failure_classification=runtime")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
