#!/usr/bin/env python3
"""Prepare or validate a pinned SU4Q N300 conformance session."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tt_rqm_kernels.su4q_conformance import prepare_session, validate_session


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--items", type=int, choices=(128, 4096), required=True)
    prepare.add_argument("--compiler-root", type=Path, required=True)
    prepare.add_argument("--entanglement-root", type=Path, required=True)
    prepare.add_argument("--tt-rqm-root", type=Path, required=True)
    prepare.add_argument("--tt-metal-root", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--session-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        path = prepare_session(
            args.output_dir.resolve(),
            items=args.items,
            compiler_root=args.compiler_root,
            entanglement_root=args.entanglement_root,
            tt_rqm_root=args.tt_rqm_root,
            tt_metal_root=args.tt_metal_root,
        )
        print(path)
        return 0
    result = validate_session(args.session_dir.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
