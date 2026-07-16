#!/usr/bin/env python3
"""Generate the clean H2A reproduction evidence hash inventory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tt_rqm_kernels.hamiltonian_lowering_evidence import (
    CLEAN_REPRODUCTION_INDEX,
    build_clean_reproduction_index,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    output = root / CLEAN_REPRODUCTION_INDEX
    output.write_text(
        json.dumps(build_clean_reproduction_index(root), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
