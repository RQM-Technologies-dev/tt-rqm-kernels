#!/usr/bin/env python3
"""Validate committed SU2ComposeBench evidence or collect a new isolated session."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tt_rqm_kernels.su2_benchmark import load_su2_preregistration
from tt_rqm_kernels.backends.tenstorrent.su2_compose_persistent import (
    render_su2_markdown,
    run_su2_compose,
)
from tt_rqm_kernels.su2_benchmark_release import validate_release


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
    performance = validate_release(REPO_ROOT / "benchmarks/manifests/wormhole-su2-compose.json")
    print(
        "SU2ComposeBench releases valid: Claim Level 0 conformance and "
        f"Claim Level {performance['claim']['level']} comparison, stable_benchmark=false"
    )


def _collect(command: str, stage: str, methodology_note: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    directory = REPO_ROOT / "benchmarks/raw/su2-compose" / timestamp
    directory.mkdir(parents=True, exist_ok=False)
    report = run_su2_compose(command=command, stage=stage, methodology_note=methodology_note)
    json_path = directory / f"{stage}-report.json"
    markdown_path = directory / f"{stage}-report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(render_su2_markdown(report))
    (directory / "session.json").write_text(
        json.dumps(
            {
                "schema": "tt-rqm-su2-compose-collection.v1",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "stage": stage,
                "report": json_path.name,
                "protected_reports_overwritten": False,
                "stable_benchmark": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return directory


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--collect", choices=("conformance", "performance"))
    parser.add_argument("--command")
    parser.add_argument("--methodology-note", default="New isolated SU2ComposeBench hardware collection.")
    args = parser.parse_args()
    if args.check:
        _check()
    else:
        if not args.command:
            parser.error("--collect requires --command")
        print(_collect(args.command, args.collect, args.methodology_note))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
