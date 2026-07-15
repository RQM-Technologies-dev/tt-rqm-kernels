#!/usr/bin/env python3
"""Validate the retained N300 SU2ComposeBench candidate experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tt_rqm_kernels.su2_candidate_experiment import (
    DEFAULT_MANIFEST,
    SU2CandidateExperimentError,
    validate_candidate_experiment,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    try:
        manifest = validate_candidate_experiment(args.manifest, repo_root=REPO_ROOT)
    except SU2CandidateExperimentError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(
        f"valid {manifest['schema']}: {manifest['experiment_id']} "
        "(diagnostic, stable_benchmark=false)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
