#!/usr/bin/env python3
"""Assess three retained non-designated SU2ComposeBench v3 pilots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tt_rqm_kernels.su2_stability import assess_v3_pilots, write_qualification


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifests", type=Path, nargs=3)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=Path("benchmarks/manifests/su2-compose-stability-preregistration-v3.json"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    assessment = assess_v3_pilots(
        args.manifests,
        preregistration_path=args.preregistration,
        repo_root=args.repo_root,
    )
    if args.output:
        write_qualification(args.output, assessment)
    print(json.dumps(assessment, indent=2, sort_keys=True))
    return 0 if assessment["ready_to_freeze_v3"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
