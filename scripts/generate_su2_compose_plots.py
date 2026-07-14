#!/usr/bin/env python3
"""Regenerate deterministic SU2ComposeBench processed data and SVGs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tt_rqm_kernels.su2_benchmark_release import DEFAULT_MANIFEST, generate_release


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    for path in generate_release(args.manifest):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
