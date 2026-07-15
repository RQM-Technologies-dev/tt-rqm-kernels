#!/usr/bin/env python3
"""Collect one non-designated fused-only SU2ComposeBench v3 pilot."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shlex
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tt_rqm_kernels.backends.tenstorrent.su2_compose_persistent import (
    FUSED_STABILITY,
    fused_stability_case_specs,
)
from tt_rqm_kernels.hardware_session import sha256_file
from tt_rqm_kernels.su2_hardware_session import collect_su2_session, parse_cpu_affinity
from tt_rqm_kernels.su2_stability import (
    load_stability_preregistration,
    load_v3_pilot_repeat_counts,
)


def _git_head(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", required=True)
    parser.add_argument("--session-id")
    parser.add_argument("--repository-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--tt-metal-root", type=Path, default=REPO_ROOT.parent / "tt-metal")
    parser.add_argument("--tt-smi-command", default="tt-smi -s")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "benchmarks/pilots/su2-compose-v3")
    parser.add_argument("--cpu-affinity", help="Candidate CPU list, for example 2-5 or 2,4,6")
    parser.add_argument(
        "--conformance-only",
        action="store_true",
        help="Run the two fused-only correctness cases without pilot timing collection.",
    )
    parser.add_argument(
        "--repeat-counts",
        type=Path,
        default=Path("benchmarks/manifests/su2-compose-v3-pilot-repeat-counts.json"),
    )
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=Path("benchmarks/manifests/su2-compose-stability-preregistration-v3.json"),
    )
    args = parser.parse_args()

    repository_root = args.repository_root.resolve()
    tt_metal_root = args.tt_metal_root.resolve()
    preregistration = load_stability_preregistration(
        args.preregistration, repo_root=repository_root
    )
    if preregistration["status"] != "pilot_foundation_not_frozen":
        parser.error("pilot collector requires the explicitly non-frozen v3 foundation")
    repeat_counts = (
        None
        if args.conformance_only
        else load_v3_pilot_repeat_counts(args.repeat_counts, repo_root=repository_root)
    )
    candidate_tokens = shlex.split(args.command)
    if len(candidate_tokens) != 1:
        parser.error("--command must name one direct candidate executable")
    candidate = Path(candidate_tokens[0]).resolve()
    prefix = "conformance" if args.conformance_only else "pilot"
    session_id = args.session_id or datetime.now(timezone.utc).strftime(
        f"{prefix}-%Y%m%dT%H%M%S.%fZ"
    )
    destination = args.output_root.resolve() / session_id
    result = collect_su2_session(
        session_dir=destination,
        session_id=session_id,
        command=str(candidate),
        methodology_note=(
            "Non-designated SU2ComposeBench Level 2 v3 fused-only conformance; isolated "
            "runtime cache and no timing or stability claim."
            if args.conformance_only
            else "Non-designated SU2ComposeBench Level 2 v3 pilot; fused_stability only, "
            "five warmups, ten raw-duration samples, isolated runtime cache, and no claim."
        ),
        repository_root=repository_root,
        tt_metal_root=tt_metal_root,
        expected_candidate_sha256=sha256_file(candidate),
        expected_execution_source_commit=_git_head(repository_root),
        expected_tt_metal_commit=_git_head(tt_metal_root),
        stability_preregistration=str(args.preregistration),
        invocation=shlex.join(sys.argv),
        tt_smi_command=args.tt_smi_command,
        benchmark_mode=FUSED_STABILITY,
        benchmark_stage="conformance" if args.conformance_only else "performance",
        case_specs=None if repeat_counts is None else fused_stability_case_specs(repeat_counts),
        designated_stability_session=False,
        isolate_runtime_cache=True,
        cpu_affinity=parse_cpu_affinity(args.cpu_affinity),
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
