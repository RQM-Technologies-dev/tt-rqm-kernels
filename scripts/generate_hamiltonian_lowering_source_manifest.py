#!/usr/bin/env python3
"""Generate the deterministic source manifest for a clean H2A candidate."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tt_rqm_kernels.hamiltonian_lowering_source_identity import (
    build_source_manifest,
    write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_source_manifest(args.repo_root, require_clean=True)
    write_json(args.output, manifest)
    print(f"repository_commit={manifest['repository_commit']}")
    print(f"source_bundle_sha256={manifest['source_bundle_sha256']}")
    print(f"file_count={len(manifest['files'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
