#!/usr/bin/env python3
"""Validate the frozen H2A designated manifest and inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tt_rqm_kernels.hamiltonian_lowering_designated import MANIFEST_PATH, validate_designated_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    manifest = (args.manifest or root / MANIFEST_PATH).resolve()
    print(json.dumps(validate_designated_manifest(manifest, root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
