#!/usr/bin/env python3
"""Regenerate or check the public H2A conformance summary."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tt_rqm_kernels.hamiltonian_lowering_release import (
    RELEASE_MANIFEST_PATH,
    generate_release,
    validate_release,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        release = validate_release(RELEASE_MANIFEST_PATH, repo_root=REPO)
        print(
            f"H2A release reproducible: Claim Level {release['claim']['level']}, "
            "stable_benchmark=false"
        )
    else:
        for output in generate_release(RELEASE_MANIFEST_PATH, repo_root=REPO):
            print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
