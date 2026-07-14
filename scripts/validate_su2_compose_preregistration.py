#!/usr/bin/env python3
"""Validate the committed SU2ComposeBench preregistration."""

from __future__ import annotations

import argparse
from pathlib import Path

from tt_rqm_kernels.su2_benchmark import load_su2_preregistration


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("benchmarks/manifests/su2-compose-preregistration.json"),
    )
    args = parser.parse_args()
    manifest = load_su2_preregistration(args.manifest)
    print(
        "SU2ComposeBench preregistration valid "
        f"({len(manifest['performance_cases'])} performance cases, status={manifest['status']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
