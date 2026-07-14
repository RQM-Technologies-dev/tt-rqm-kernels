#!/usr/bin/env python3
"""Validate frozen SU2ComposeBench numerical stability gates."""

from __future__ import annotations

import argparse
from pathlib import Path

from tt_rqm_kernels.su2_stability import load_stability_preregistration


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("benchmarks/manifests/su2-compose-stability-preregistration.json"),
    )
    args = parser.parse_args()
    preregistration = load_stability_preregistration(args.manifest)
    print(
        "SU2ComposeBench stability preregistration valid "
        f"({len(preregistration['cases'])} cases, "
        f"sessions={preregistration['session_contract']['required_designated_sessions']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
