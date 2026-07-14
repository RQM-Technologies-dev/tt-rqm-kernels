#!/usr/bin/env python3
"""Validate SU2ComposeBench evidence, claims, provenance, and generated outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tt_rqm_kernels.su2_benchmark_release import DEFAULT_MANIFEST, SU2ReleaseError, validate_release


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--skip-generated", action="store_true")
    args = parser.parse_args()
    try:
        release = validate_release(args.manifest, verify_generated=not args.skip_generated)
    except SU2ReleaseError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"valid {release['schema']}: {release['benchmark_id']} (Claim Level {release['claim']['level']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
