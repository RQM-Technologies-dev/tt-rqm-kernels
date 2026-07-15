#!/usr/bin/env python3
"""Run one exact SU2ComposeBench profiler case in one candidate process."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tt_rqm_kernels.backends.tenstorrent.su2_compose_persistent import (
    render_su2_markdown,
    run_su2_compose,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--candidate-stdout", type=Path, required=True)
    parser.add_argument("--candidate-stderr", type=Path, required=True)
    args = parser.parse_args()
    capture: dict[str, str] = {}
    try:
        report = run_su2_compose(
            command=args.command,
            stage="profile",
            methodology_note=(
                "Diagnostic Device Program Profiler and Tracy capture; one fused/unfused pair, "
                "no timing warmups, no stability or acceleration claim."
            ),
            expected_candidate_sha256=args.expected_candidate_sha256,
            expected_repository_commit=args.expected_source_commit,
            process_capture=capture,
            case_specs=((args.batch, args.steps, 1, 0, 1),),
        )
    finally:
        args.candidate_stdout.write_text(capture.get("stdout", ""))
        args.candidate_stderr.write_text(capture.get("stderr", ""))
    args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.markdown_output.write_text(render_su2_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
