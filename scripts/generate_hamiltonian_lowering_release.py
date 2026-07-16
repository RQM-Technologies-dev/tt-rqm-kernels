#!/usr/bin/env python3
"""Generate the hash-bound public H2A Claim Level 0 manifest and summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tt_rqm_kernels.hamiltonian_lowering_release import (
    RELEASE_MANIFEST_PATH,
    build_release_manifest,
    generate_release,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    manifest_path = root / RELEASE_MANIFEST_PATH
    manifest_path.write_text(
        json.dumps(build_release_manifest(root), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    generate_release(manifest_path, repo_root=root)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
