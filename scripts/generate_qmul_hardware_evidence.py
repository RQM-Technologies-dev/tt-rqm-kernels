#!/usr/bin/env python3
"""Generate deterministic processed reports from isolated qmul evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from tt_rqm_kernels.qmul_hardware_evidence import generate_all


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    for path in generate_all(args.repo_root):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
