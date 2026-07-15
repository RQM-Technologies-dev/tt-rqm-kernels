#!/usr/bin/env python3
"""Collect the exact four-case SU2ComposeBench profiler session."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tt_rqm_kernels.su2_profile import SU2ProfileError, collect_profile_session


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--collector-root", type=Path, required=True)
    parser.add_argument("--execution-source-root", type=Path, required=True)
    parser.add_argument("--tt-metal-root", type=Path, required=True)
    parser.add_argument("--tracy-capture", type=Path, required=True)
    parser.add_argument("--tracy-csvexport", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    args = parser.parse_args()
    try:
        output = collect_profile_session(
            candidate=args.candidate,
            collector_root=args.collector_root,
            execution_source_root=args.execution_source_root,
            tt_metal_root=args.tt_metal_root,
            tracy_capture=args.tracy_capture,
            tracy_csvexport=args.tracy_csvexport,
            output_root=args.output_root,
            session_id=args.session_id,
        )
    except SU2ProfileError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
