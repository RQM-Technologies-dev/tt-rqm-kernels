#!/usr/bin/env python3
"""Check the public release or collect a new isolated hardware session."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import sys

from tt_rqm_kernels.benchmark_release import (
    BenchmarkReleaseError,
    DEFAULT_MANIFEST,
    validate_release,
)
from tt_rqm_kernels.hardware_session import collect_qmul_session


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the published Wormhole qmul release or collect a new "
            "device-0 persistent session without overwriting canonical reports."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument(
        "--collect-stage",
        choices=("conformance", "performance", "diagnostic"),
        help="Run an explicit hardware collection into a new timestamped directory.",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--command", help="Real Wormhole persistent candidate executable.")
    parser.add_argument("--output-root", type=Path, default=Path("benchmarks/raw"))
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device-id", type=int, choices=(0, 1), default=0)
    parser.add_argument("--output-cb-depth", type=int, choices=(2, 4), default=2)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--tt-metal-root", type=Path)
    parser.add_argument("--tt-smi-command", default="tt-smi -s")
    parser.add_argument("--runtime-version", default="TT-Metalium 0.75.0")
    parser.add_argument("--expected-candidate-sha256")
    parser.add_argument("--expected-execution-source-commit")
    parser.add_argument("--expected-tt-metal-commit")
    parser.add_argument("--source-tree-sha256")
    parser.add_argument(
        "--case-spec",
        action="append",
        default=[],
        help="Diagnostic case as N,iterations,warmup,samples[,requested_max_cores].",
    )
    args = parser.parse_args()

    if args.check:
        try:
            manifest = validate_release(args.manifest)
        except BenchmarkReleaseError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(
            f"benchmark release valid: {manifest['benchmark_id']} "
            f"(Claim Level {manifest['claim']['level']}, "
            f"stable_benchmark={str(manifest['claim']['stable_benchmark']).lower()})"
        )
        return 0

    if not args.command:
        parser.error("--command is required with --collect-stage")
    if not args.session_id:
        parser.error("--session-id is required for designated hardware collection")
    if args.tt_metal_root is None:
        parser.error("--tt-metal-root is required for designated hardware collection")
    session_id = args.session_id
    if not session_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_." for character in session_id):
        parser.error("--session-id may contain only letters, numbers, dash, underscore, and dot")
    session_dir = args.output_root / session_id
    try:
        reference = validate_release(
            args.manifest, repo_root=args.repository_root, verify_generated=False
        )
        provenance = reference["provenance"]
        case_specs = []
        for value in args.case_spec:
            try:
                parsed = tuple(int(part) for part in value.split(","))
            except ValueError as exc:
                parser.error(f"invalid --case-spec {value!r}: {exc}")
            if len(parsed) not in (4, 5):
                parser.error("--case-spec needs 4 or 5 comma-separated integers")
            case_specs.append(parsed)
        note = (
            f"Independent persistent Wormhole device-{args.device_id} {args.collect_stage} "
            f"session {session_id}; stability qualification is separate."
        )
        collect_qmul_session(
            session_dir=session_dir,
            session_id=session_id,
            command=args.command,
            benchmark_stage=args.collect_stage,
            methodology_note=note,
            repository_root=args.repository_root,
            tt_metal_root=args.tt_metal_root,
            expected_candidate_sha256=(args.expected_candidate_sha256 or provenance["candidate_sha256"]),
            expected_execution_source_commit=(args.expected_execution_source_commit or provenance["repository_commit"]),
            expected_tt_metal_commit=(args.expected_tt_metal_commit or provenance["tt_metal_commit"]),
            device_id=args.device_id,
            output_cb_depth=args.output_cb_depth,
            seed=args.seed,
            invocation=shlex.join([sys.executable, *sys.argv]),
            tt_smi_command=args.tt_smi_command,
            runtime_version=args.runtime_version,
            case_specs=case_specs or None,
            source_tree_sha256=args.source_tree_sha256,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(session_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
