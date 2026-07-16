#!/usr/bin/env python3
"""Serialize the exact H2A designated input packages."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tt_rqm_kernels.hamiltonian_lowering_designated import freeze_inputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = freeze_inputs(args.output_root)
    print(f"case_count={len(result['cases'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
