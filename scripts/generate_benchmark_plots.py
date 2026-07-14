#!/usr/bin/env python3
"""Regenerate committed benchmark summaries and SVG plots."""

from __future__ import annotations

import argparse
from pathlib import Path

from tt_rqm_kernels.benchmark_release import DEFAULT_MANIFEST, generate_release


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    generated = generate_release(args.manifest)
    for path in generated:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
