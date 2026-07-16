#!/usr/bin/env python3
"""Qualify a retained H2B pilot package entirely offline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tt_rqm_kernels.hamiltonian_evolution_pilot import (
    HamiltonianEvolutionPilotError,
    build_qualification,
    render_qualification,
    validate_pilot_package,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "package",
        type=Path,
        nargs="?",
        default=ROOT
        / "benchmarks/pilots/hamiltonian-evolution-h2b/h2b-n300-pilot-20260716-session-1",
    )
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmarks/processed/hamiltonian-evolution-h2b-pilot-qualification.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "reports/hamiltonian-evolution-h2b-pilot.md",
    )
    args = parser.parse_args()
    try:
        result = validate_pilot_package(args.package, ROOT)
        qualification = build_qualification(args.package, ROOT)
    except HamiltonianEvolutionPilotError as exc:
        print(f"H2B pilot package invalid: {exc}")
        return 1
    expected_json = json.dumps(qualification, indent=2, sort_keys=True) + "\n"
    expected_report = render_qualification(qualification)
    if args.check:
        if not args.output.is_file() or args.output.read_text(encoding="utf-8") != expected_json:
            print("H2B pilot processed qualification is missing or stale")
            return 1
        if not args.report.is_file() or args.report.read_text(encoding="utf-8") != expected_report:
            print("H2B pilot qualification report is missing or stale")
            return 1
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(expected_json, encoding="utf-8")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(expected_report, encoding="utf-8")
    print(f"package_valid={str(result['package_valid']).lower()}")
    print(f"pilot_passed={str(result['pilot_passed']).lower()}")
    print(f"case_count={result['case_count']}")
    return 0 if result["pilot_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
