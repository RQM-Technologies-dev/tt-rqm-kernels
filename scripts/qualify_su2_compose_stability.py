#!/usr/bin/env python3
"""Qualify three designated SU2ComposeBench cold-start sessions."""

from __future__ import annotations

import argparse
from pathlib import Path

from tt_rqm_kernels.su2_stability import qualify_stability, write_qualification


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("session_manifests", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=Path("benchmarks/manifests/su2-compose-stability-preregistration.json"),
    )
    parser.add_argument(
        "--expected-outcome",
        choices=("passed", "failed"),
        default="passed",
        help="Expected deterministic qualification outcome (default: passed).",
    )
    args = parser.parse_args()
    qualification = qualify_stability(
        args.session_manifests,
        preregistration_path=args.preregistration,
    )
    write_qualification(args.output, qualification)
    print(args.output)
    print(f"qualification_passed={str(qualification['qualification_passed']).lower()}")
    expected_passed = args.expected_outcome == "passed"
    return 0 if qualification["qualification_passed"] is expected_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
