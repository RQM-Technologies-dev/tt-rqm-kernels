#!/usr/bin/env python3
"""Qualify three designated persistent qmul cold-start sessions."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tt_rqm_kernels.qmul_stability import qualify_stability, write_qualification


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("session_manifests", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    qualification = qualify_stability(args.session_manifests)
    write_qualification(args.output, qualification)
    print(args.output)
    print(f"stable_benchmark={str(qualification['stable_benchmark']).lower()}")
    return 0 if qualification["stable_benchmark"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
