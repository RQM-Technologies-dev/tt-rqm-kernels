#!/usr/bin/env python3
"""Prepare, host-check, or validate a chained SU4Q conformance session."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tt_rqm_kernels.su4q_chain_conformance import (
    DEPTHS,
    host_candidate,
    prepare_session,
    validate_session,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--depth", type=int, choices=DEPTHS, required=True)
    prepare.add_argument("--compiler-root", type=Path, required=True)
    prepare.add_argument("--entanglement-root", type=Path, required=True)
    prepare.add_argument("--tt-rqm-root", type=Path, required=True)
    prepare.add_argument("--tt-metal-root", type=Path, required=True)
    host = commands.add_parser("host-check")
    host.add_argument("--session-dir", type=Path, required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--session-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        path = prepare_session(
            args.output_dir.resolve(),
            depth=args.depth,
            compiler_root=args.compiler_root,
            entanglement_root=args.entanglement_root,
            tt_rqm_root=args.tt_rqm_root,
            tt_metal_root=args.tt_metal_root,
        )
        print(path)
        return 0
    if args.command == "host-check":
        path = host_candidate(args.session_dir.resolve())
        result = validate_session(
            args.session_dir.resolve(), output_name=path.name
        )
    else:
        result = validate_session(args.session_dir.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
