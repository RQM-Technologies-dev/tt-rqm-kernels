#!/usr/bin/env python3
"""Validate retained SU2ComposeBench profiler attempts and attribution."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tt_rqm_kernels.su2_profile import (
    DEFAULT_EVIDENCE_MANIFEST,
    SU2ProfileError,
    validate_profile_evidence,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_EVIDENCE_MANIFEST)
    args = parser.parse_args()
    try:
        manifest = validate_profile_evidence(args.manifest, repo_root=REPO_ROOT)
    except (OSError, ValueError, SU2ProfileError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(
        f"valid {manifest['schema']}: {len(manifest['attempts'])} retained attempts, "
        "exact candidate retained, stable_benchmark=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
