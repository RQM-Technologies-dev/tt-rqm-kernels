#!/usr/bin/env python3
"""Validate the public H2A Claim Level 0 release entirely offline."""

from __future__ import annotations

from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tt_rqm_kernels.hamiltonian_lowering_release import RELEASE_MANIFEST_PATH, validate_release


if __name__ == "__main__":
    release = validate_release(RELEASE_MANIFEST_PATH, repo_root=REPO)
    print(
        f"H2A release valid: {release['benchmark_id']} "
        f"(Claim Level {release['claim']['level']}, stable_benchmark=false)"
    )
