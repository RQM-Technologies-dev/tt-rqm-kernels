#!/usr/bin/env python3
"""Validate committed SU2ComposeBench evidence or collect a new isolated session."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tt_rqm_kernels.su2_benchmark import load_su2_preregistration
from tt_rqm_kernels.su2_benchmark_release import (
    published_manifest_path,
    validate_release,
)
from tt_rqm_kernels.su2_hardware_session import collect_su2_session
from tt_rqm_kernels.su2_stability import (
    load_stability_preregistration,
    load_v3_pilot_repeat_counts,
)


FROZEN_CANDIDATE_SHA256 = "d8237f2e5b05885167085d87a0400daf8b5feb0318d906285a1d263035294441"
FROZEN_EXECUTION_SOURCE = "3238299a9eea2a44dccd6826a947cac3266dd2f7"
FROZEN_TT_METAL = "dd2849b5bc6b7a5d38a9eafbeba31ef8d530f8d4"
V2_PREREGISTRATION = Path("benchmarks/manifests/su2-compose-stability-preregistration-v2.json")
V3_PREREGISTRATION = Path("benchmarks/manifests/su2-compose-stability-preregistration-v3.json")


def _check() -> None:
    load_su2_preregistration(REPO_ROOT / "benchmarks/manifests/su2-compose-preregistration.json")
    load_stability_preregistration(repo_root=REPO_ROOT)
    if (REPO_ROOT / V2_PREREGISTRATION).is_file():
        load_stability_preregistration(V2_PREREGISTRATION, repo_root=REPO_ROOT)
    if (REPO_ROOT / V3_PREREGISTRATION).is_file():
        v3 = load_stability_preregistration(V3_PREREGISTRATION, repo_root=REPO_ROOT)
        if v3["status"] != "pilot_foundation_not_frozen":
            raise ValueError("unexpected v3 campaign status")
        load_v3_pilot_repeat_counts(
            Path("benchmarks/manifests/su2-compose-v3-pilot-repeat-counts.json"),
            repo_root=REPO_ROOT,
        )
    release_path = REPO_ROOT / "benchmarks/manifests/su2-compose-conformance.json"
    release = json.loads(release_path.read_text())
    if release.get("schema") != "tt-rqm-su2-compose-conformance-release.v1":
        raise ValueError("unsupported conformance release schema")
    if release.get("claim") != {
        "level": 0,
        "name": "silicon_conformance",
        "stable_benchmark": False,
    }:
        raise ValueError("conformance release claim mismatch")
    for artifact in release.get("artifacts", []):
        path = REPO_ROOT / artifact["path"]
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != artifact["sha256"]:
            raise ValueError(f"artifact hash mismatch: {artifact['path']}")
    report = json.loads(
        (REPO_ROOT / "reports/tt_hardware_su2_compose_conformance.json").read_text()
    )
    if (
        report.get("benchmark_stage") != "conformance"
        or report.get("performance_eligible") is not False
    ):
        raise ValueError("conformance report eligibility mismatch")
    if report.get("stable_benchmark") is not False or report.get("lifecycle") != {
        "close_count": 1,
        "create_count": 1,
        "device_count": 1,
        "device_id": 0,
    }:
        raise ValueError("conformance report lifecycle mismatch")
    for result in report.get("results", []):
        for path in ("fused", "unfused"):
            correctness = result[path]["correctness"]
            if correctness["failing_values"] or correctness["nonfinite_values"]:
                raise ValueError("conformance report contains correctness failures")
            if correctness["max_abs_error"] > 1e-4:
                raise ValueError("conformance report exceeds tolerance")
    performance = validate_release(
        REPO_ROOT / published_manifest_path(REPO_ROOT),
        repo_root=REPO_ROOT,
    )
    print(
        "SU2ComposeBench releases valid: Claim Level 0 conformance and "
        f"Claim Level {performance['claim']['level']} comparison, "
        f"stable_benchmark={str(performance['claim']['stable_benchmark']).lower()}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--collect", choices=("performance",))
    parser.add_argument("--command")
    parser.add_argument("--session-id")
    parser.add_argument("--repository-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--tt-metal-root", type=Path, default=REPO_ROOT.parent / "tt-metal")
    parser.add_argument("--tt-smi-command", default="tt-smi -s")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--execution-source-root", type=Path)
    parser.add_argument(
        "--stability-preregistration",
        type=Path,
        default=Path("benchmarks/manifests/su2-compose-stability-preregistration.json"),
    )
    parser.add_argument(
        "--methodology-note",
        default=(
            "Designated SU2ComposeBench stability session; frozen eight-case order, "
            "two warmup pairs, ten measured pairs, and no discarded runs."
        ),
    )
    args = parser.parse_args()
    if args.check:
        _check()
    else:
        if not args.command:
            parser.error("--collect requires --command")
        preregistration = load_stability_preregistration(
            args.stability_preregistration, repo_root=args.repository_root
        )
        candidate = preregistration.get("candidate", {})
        candidate_sha256 = candidate.get("sha256", FROZEN_CANDIDATE_SHA256)
        source_commit = candidate.get("source_commit", FROZEN_EXECUTION_SOURCE)
        tt_metal_commit = candidate.get("tt_metal_commit", FROZEN_TT_METAL)
        output_root = args.output_root or (REPO_ROOT / "benchmarks/raw/su2-compose")
        session_id = args.session_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        directory = output_root / session_id
        print(
            collect_su2_session(
                session_dir=directory,
                session_id=session_id,
                command=args.command,
                methodology_note=args.methodology_note,
                repository_root=args.repository_root,
                tt_metal_root=args.tt_metal_root,
                expected_candidate_sha256=str(candidate_sha256),
                expected_execution_source_commit=str(source_commit),
                expected_tt_metal_commit=str(tt_metal_commit),
                expected_compiler_version=candidate.get("compiler_version"),
                expected_runtime_version=candidate.get("runtime_version"),
                execution_source_root=args.execution_source_root,
                expected_source_tree_sha256=candidate.get("source_tree_sha256"),
                stability_preregistration=str(args.stability_preregistration),
                invocation=shlex.join(sys.argv),
                tt_smi_command=args.tt_smi_command,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
