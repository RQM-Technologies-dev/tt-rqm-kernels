#!/usr/bin/env python3
"""Check the public release or collect a new isolated hardware session."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from tt_rqm_kernels.benchmark_release import (
    BenchmarkReleaseError,
    DEFAULT_MANIFEST,
    sha256_file,
    validate_release,
)
from tt_rqm_kernels.backends.tenstorrent.qmul_persistent import (
    render_persistent_markdown,
    run_persistent_qmul,
)


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
        choices=("conformance", "performance"),
        help="Run an explicit hardware collection into a new timestamped directory.",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--command", help="Real Wormhole persistent candidate executable.")
    parser.add_argument("--output-root", type=Path, default=Path("benchmarks/raw"))
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--seed", type=int, default=0)
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
    session_id = args.session_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not session_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_." for character in session_id):
        parser.error("--session-id may contain only letters, numbers, dash, underscore, and dot")
    session_dir = args.output_root / f"wormhole-qmul-{session_id}"
    try:
        session_dir.mkdir(parents=True, exist_ok=False)
        note = (
            f"Independent public persistent Wormhole device-0 {args.collect_stage} "
            f"session {session_id}; stability qualification is separate."
        )
        report = run_persistent_qmul(
            command=args.command,
            benchmark_stage=args.collect_stage,
            methodology_note=note,
            seed=args.seed,
        )
        json_path = session_dir / "report.json"
        markdown_path = session_dir / "report.md"
        json_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        markdown_path.write_text(render_persistent_markdown(report), encoding="utf-8")
        session_manifest = {
            "schema": "tt-rqm-benchmark-session.v1",
            "session_id": session_id,
            "benchmark_stage": args.collect_stage,
            "device_count": 1,
            "device_id": 0,
            "stable_benchmark": False,
            "report": {"path": "report.json", "sha256": sha256_file(json_path)},
            "markdown": {"path": "report.md", "sha256": sha256_file(markdown_path)},
        }
        (session_dir / "session-manifest.json").write_text(
            json.dumps(session_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(session_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
