#!/usr/bin/env python3
"""Fail-closed collector retained for the frozen H2A designated contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tt_rqm_kernels.hamiltonian_lowering_designated import (
    MANIFEST_PATH,
    collect_designated_session,
    dry_run_preflight,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=REPO / MANIFEST_PATH)
    parser.add_argument("--governance-root", type=Path, default=REPO)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--tt-metal-root", type=Path, required=True)
    parser.add_argument("--candidate-binary", type=Path, required=True)
    parser.add_argument("--tt-smi", default="tt-smi")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--session-id")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    common = {
        "manifest_path": args.manifest.resolve(),
        "governance_root": args.governance_root.resolve(),
        "source_repo": args.source_repo.resolve(),
        "tt_metal_root": args.tt_metal_root.resolve(),
        "candidate_binary": args.candidate_binary.resolve(),
        "tt_smi": args.tt_smi,
    }
    if args.dry_run:
        if args.session_id is not None or args.output_root is not None:
            parser.error("dry-run forbids session identity and output")
        result = dry_run_preflight(**common)
    else:
        if args.session_id is None or args.output_root is None:
            parser.error("designated execution requires --session-id and --output-root")
        result = collect_designated_session(
            **common, session_id=args.session_id, output_root=args.output_root.resolve()
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
