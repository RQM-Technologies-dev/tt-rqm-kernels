#!/usr/bin/env python3
"""Generate or validate the deterministic H2B source inventory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tt_rqm_kernels.hamiltonian_evolution_source_identity import (
    validate_source_manifest,
    write_source_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "benchmarks/manifests/hamiltonian-evolution-h2b-source-manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        payload = validate_source_manifest(args.output, ROOT)
    else:
        payload = write_source_manifest(args.output, ROOT)
    print(
        json.dumps(
            {
                "repository_commit": payload["repository_commit"],
                "source_bundle_sha256": payload["source_bundle_sha256"],
                "file_count": payload["file_count"],
                "source_scope_clean": payload["source_scope_clean"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
