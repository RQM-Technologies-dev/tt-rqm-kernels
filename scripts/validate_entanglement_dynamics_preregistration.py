#!/usr/bin/env python3
"""Validate the committed EntanglementDynamicsBench preregistration."""

from __future__ import annotations

import argparse
from pathlib import Path

from tt_rqm_kernels.entanglement_benchmark import load_entanglement_preregistration


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("benchmarks/manifests/entanglement-dynamics-preregistration.json"),
    )
    args = parser.parse_args()
    manifest = load_entanglement_preregistration(args.manifest)
    print(
        "EntanglementDynamicsBench preregistration valid "
        f"({len(manifest['cases'])} CPU reference cases, status={manifest['status']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
