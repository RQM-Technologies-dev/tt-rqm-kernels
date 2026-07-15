#!/usr/bin/env python3
"""Collect one immutable designated SU2ComposeBench v3 Level 2 session."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tt_rqm_kernels.backends.tenstorrent.su2_compose_persistent import (
    FUSED_STABILITY,
    fused_stability_case_specs,
)
from tt_rqm_kernels.benchmark_integrity import IntegrityError
from tt_rqm_kernels.hardware_session import sha256_file, validate_device_health
from tt_rqm_kernels.su2_hardware_session import (
    DISALLOWED_TIMING_ENV,
    capture_host_state,
    collect_su2_session,
)
from tt_rqm_kernels.su2_profile import validate_source_tree
from tt_rqm_kernels.su2_stability import (
    V3_HOST_IDENTITY_KEYS,
    load_stability_preregistration,
    load_v3_pilot_repeat_counts,
)


PREREGISTRATION = Path("benchmarks/manifests/su2-compose-stability-preregistration-v3.json")
DESIGNATED_SESSION_IDS = (
    "su2-v3-level2-session-1",
    "su2-v3-level2-session-2",
    "su2-v3-level2-session-3",
)
FROZEN_CPU_AFFINITY = frozenset({24, 25, 26, 27})


def _git_value(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _compiler_version() -> str:
    completed = subprocess.run(["c++", "--version"], check=True, capture_output=True, text=True)
    output = (completed.stdout or completed.stderr).strip()
    if not output:
        raise IntegrityError("C++ compiler produced no version output")
    return output.splitlines()[0]


def _timing_environment() -> dict[str, str]:
    return {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith("TT_METAL_") or key.startswith("TT_RQM_")
    }


def _assert_host_contract(host: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    for key in V3_HOST_IDENTITY_KEYS:
        if host.get(key) != contract.get(key):
            raise IntegrityError(f"host contract differs for {key}")


def preflight(
    *,
    command: str,
    repository_root: Path,
    execution_source_root: Path,
    tt_metal_root: Path,
    tt_smi_command: str,
    preregistration_path: Path,
) -> dict[str, Any]:
    """Validate every frozen identity without creating a designated session."""

    repository_root = repository_root.resolve()
    execution_source_root = execution_source_root.resolve()
    tt_metal_root = tt_metal_root.resolve()
    preregistration = load_stability_preregistration(preregistration_path, repo_root=repository_root)
    if preregistration.get("status") != "frozen_before_designated_session_1":
        raise IntegrityError("designated collection requires the frozen v3 preregistration")
    candidate = preregistration["candidate"]
    tokens = shlex.split(command)
    if len(tokens) != 1:
        raise IntegrityError("designated collection requires one direct candidate executable")
    candidate_path = Path(tokens[0]).resolve()
    if not candidate_path.is_file() or not os.access(candidate_path, os.X_OK):
        raise IntegrityError(f"candidate is not executable: {candidate_path}")
    if sha256_file(candidate_path) != candidate["sha256"]:
        raise IntegrityError("candidate SHA-256 differs from frozen v3 identity")
    if _git_value(repository_root, "status", "--porcelain"):
        raise IntegrityError("collector repository is dirty")
    if _git_value(tt_metal_root, "status", "--porcelain"):
        raise IntegrityError("TT-Metal source tree is dirty")
    if _git_value(tt_metal_root, "rev-parse", "HEAD") != candidate["tt_metal_commit"]:
        raise IntegrityError("TT-Metal commit differs from frozen v3 identity")
    source_tree_sha256 = validate_source_tree(
        collector_root=repository_root,
        source_root=execution_source_root,
        source_commit=candidate["source_commit"],
    )
    if source_tree_sha256 != candidate["source_tree_sha256"]:
        raise IntegrityError("execution source tree differs from frozen v3 identity")
    if _compiler_version() != candidate["compiler_version"]:
        raise IntegrityError("compiler version differs from frozen v3 identity")
    timing_environment = _timing_environment()
    if timing_environment != preregistration["host_contract"]["timing_environment"]:
        raise IntegrityError("timing environment differs from frozen v3 host contract")
    enabled_debug = {
        key: value
        for key, value in timing_environment.items()
        if key in DISALLOWED_TIMING_ENV and value.strip().lower() not in {"", "0", "false", "off", "no"}
    }
    if enabled_debug:
        raise IntegrityError(f"profiler, watcher, or debug environment is enabled: {enabled_debug}")
    host = capture_host_state(cpu_affinity=FROZEN_CPU_AFFINITY)
    _assert_host_contract(host, preregistration["host_contract"])
    health = subprocess.run(
        shlex.split(tt_smi_command), check=True, capture_output=True, text=True
    ).stdout
    validate_device_health(health, device_id=0)
    return {
        "candidate_sha256": candidate["sha256"],
        "source_commit": candidate["source_commit"],
        "source_tree_sha256": source_tree_sha256,
        "tt_metal_commit": candidate["tt_metal_commit"],
        "compiler_version": candidate["compiler_version"],
        "runtime_version": candidate["runtime_version"],
        "host_contract": preregistration["host_contract"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", required=True)
    parser.add_argument("--session-id", choices=DESIGNATED_SESSION_IDS)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--repository-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--execution-source-root",
        type=Path,
        help="Candidate source subtree; defaults to experimental/tt_metalium_su2_compose in the repository.",
    )
    parser.add_argument("--tt-metal-root", type=Path, default=REPO_ROOT.parent / "tt-metal")
    parser.add_argument("--tt-smi-command", default="tt-smi -s")
    parser.add_argument(
        "--output-root", type=Path, default=REPO_ROOT / "benchmarks/raw/su2-compose-v3"
    )
    parser.add_argument("--preregistration", type=Path, default=PREREGISTRATION)
    args = parser.parse_args()
    if not args.preflight and args.session_id is None:
        parser.error("--session-id is required unless --preflight is used")
    repository_root = args.repository_root.resolve()
    execution_source_root = (
        args.execution_source_root.resolve()
        if args.execution_source_root is not None
        else repository_root / "experimental/tt_metalium_su2_compose"
    )
    evidence = preflight(
        command=args.command,
        repository_root=repository_root,
        execution_source_root=execution_source_root,
        tt_metal_root=args.tt_metal_root,
        tt_smi_command=args.tt_smi_command,
        preregistration_path=args.preregistration,
    )
    if args.preflight:
        print(json.dumps(evidence, indent=2, sort_keys=True))
        return 0
    preregistration = load_stability_preregistration(args.preregistration, repo_root=repository_root)
    repeat_counts = load_v3_pilot_repeat_counts(
        Path(str(preregistration["pilot_repeat_plan"])), repo_root=repository_root
    )
    session_dir = args.output_root.resolve() / str(args.session_id)
    result = collect_su2_session(
        session_dir=session_dir,
        session_id=str(args.session_id),
        command=args.command,
        methodology_note=(
            "Designated SU2ComposeBench Level 2 v3 fused-only stability session; "
            "frozen candidate, source/runtime identity, host contract, five warmups, "
            "ten raw-duration samples, isolated runtime cache, and no individual stability claim."
        ),
        repository_root=repository_root,
        tt_metal_root=args.tt_metal_root,
        expected_candidate_sha256=str(evidence["candidate_sha256"]),
        expected_execution_source_commit=str(evidence["source_commit"]),
        expected_tt_metal_commit=str(evidence["tt_metal_commit"]),
        expected_compiler_version=str(evidence["compiler_version"]),
        expected_runtime_version=str(evidence["runtime_version"]),
        execution_source_root=execution_source_root,
        expected_source_tree_sha256=str(evidence["source_tree_sha256"]),
        stability_preregistration=str(args.preregistration),
        invocation=shlex.join(sys.argv),
        tt_smi_command=args.tt_smi_command,
        benchmark_mode=FUSED_STABILITY,
        benchmark_stage="performance",
        case_specs=fused_stability_case_specs(repeat_counts),
        designated_stability_session=True,
        isolate_runtime_cache=True,
        cpu_affinity=FROZEN_CPU_AFFINITY,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
