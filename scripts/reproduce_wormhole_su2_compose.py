#!/usr/bin/env python3
"""Validate committed SU2ComposeBench evidence or show collection guidance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tt_rqm_kernels.su2_benchmark import load_su2_preregistration


def _check() -> None:
    load_su2_preregistration(REPO_ROOT / "benchmarks/manifests/su2-compose-preregistration.json")
    release_path = REPO_ROOT / "benchmarks/manifests/su2-compose-conformance.json"
    release = json.loads(release_path.read_text())
    if release.get("schema") != "tt-rqm-su2-compose-conformance-release.v1":
        raise ValueError("unsupported conformance release schema")
    if release.get("claim") != {"level": 0, "name": "silicon_conformance", "stable_benchmark": False}:
        raise ValueError("conformance release claim mismatch")
    for artifact in release.get("artifacts", []):
        path = REPO_ROOT / artifact["path"]
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != artifact["sha256"]:
            raise ValueError(f"artifact hash mismatch: {artifact['path']}")
    report = json.loads((REPO_ROOT / "reports/tt_hardware_su2_compose_conformance.json").read_text())
    if report.get("benchmark_stage") != "conformance" or report.get("performance_eligible") is not False:
        raise ValueError("conformance report eligibility mismatch")
    if report.get("stable_benchmark") is not False or report.get("lifecycle") != {
        "close_count": 1, "create_count": 1, "device_count": 1, "device_id": 0
    }:
        raise ValueError("conformance report lifecycle mismatch")
    for result in report.get("results", []):
        for path in ("fused", "unfused"):
            correctness = result[path]["correctness"]
            if correctness["failing_values"] or correctness["nonfinite_values"]:
                raise ValueError("conformance report contains correctness failures")
            if correctness["max_abs_error"] > 1e-4:
                raise ValueError("conformance report exceeds tolerance")
    print("SU2ComposeBench release valid: Claim Level 0, stable_benchmark=false")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not args.check:
        parser.error("use --check; hardware collection is explicit through run_su2_compose_hardware.py")
    _check()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
