#!/usr/bin/env python3
"""Run the protected persistent-device Stage B qmul qualification path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from tt_rqm_kernels.backends.tenstorrent.qmul_persistent import (
    render_persistent_markdown,
    run_persistent_qmul,
)

OUTPUTS = {
    "conformance": (
        Path("reports/tt_hardware_qmul_stage_b_persistent_conformance.json"),
        Path("reports/tt_hardware_qmul_stage_b_persistent_conformance.md"),
    ),
    "performance": (
        Path("reports/tt_hardware_qmul_stage_b_persistent_performance.json"),
        Path("reports/tt_hardware_qmul_stage_b_persistent_performance.md"),
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", required=True)
    parser.add_argument("--benchmark-stage", choices=tuple(OUTPUTS), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--methodology-note", required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()
    expected_json, expected_markdown = OUTPUTS[args.benchmark_stage]
    try:
        _require_output(args.json_output, expected_json)
        _require_output(args.markdown_output, expected_markdown)
        report = run_persistent_qmul(
            command=args.command,
            benchmark_stage=args.benchmark_stage,
            methodology_note=args.methodology_note,
            seed=args.seed,
        )
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        args.markdown_output.write_text(
            render_persistent_markdown(report), encoding="utf-8"
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _require_output(observed: Path, expected: Path) -> None:
    if tuple(observed.parts[-2:]) != tuple(expected.parts):
        raise ValueError(f"protected persistent artifact path must end in {expected}")


if __name__ == "__main__":
    raise SystemExit(main())
