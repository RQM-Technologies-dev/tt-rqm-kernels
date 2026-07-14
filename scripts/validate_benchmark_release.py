#!/usr/bin/env python3
"""Validate benchmark evidence, claims, hashes, and generated files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from tt_rqm_kernels.benchmark_release import (
    BenchmarkReleaseError,
    DEFAULT_MANIFEST,
    validate_release,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--skip-generated",
        action="store_true",
        help="Validate evidence and claims without regenerating outputs.",
    )
    args = parser.parse_args()
    try:
        manifest = validate_release(
            args.manifest,
            verify_generated=not args.skip_generated,
        )
    except BenchmarkReleaseError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(
        f"valid {manifest['schema']}: {manifest['benchmark_id']} "
        f"(Claim Level {manifest['claim']['level']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
