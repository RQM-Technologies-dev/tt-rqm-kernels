#!/usr/bin/env python3
"""Offline qualifier for a future retained H2A designated session."""

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
    contract_readiness,
    qualify_session,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--contract-only", action="store_true")
    parser.add_argument("--session-root", type=Path)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    manifest = (args.manifest or root / MANIFEST_PATH).resolve()
    if args.contract_only:
        if args.session_root is not None:
            parser.error("contract-only forbids a session package")
        result = contract_readiness(manifest, root)
    else:
        if args.session_root is None:
            parser.error("qualification requires --session-root")
        result = qualify_session(args.session_root.resolve(), manifest, root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
