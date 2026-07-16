#!/usr/bin/env python3
"""Validate one external H2A coefficient-lowering candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tt_rqm_kernels.hamiltonian_lowering_candidate import (
    HamiltonianLoweringCandidateError,
    deterministic_candidate_inputs,
    run_external_candidate,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", required=True)
    parser.add_argument("--stage", choices=("conformance", "performance"), default="conformance")
    parser.add_argument(
        "--execution-label", choices=("cpu_reference", "hardware"), default="cpu_reference"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--B", type=int, default=4)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    coefficients, dt = deterministic_candidate_inputs(seed=args.seed, B=args.B, K=args.K)
    try:
        run = run_external_candidate(
            coefficients,
            dt,
            command=args.command,
            stage=args.stage,
            execution_label=args.execution_label,
        )
    except HamiltonianLoweringCandidateError as exc:
        print(str(exc))
        return 1
    rendered = json.dumps(run.report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
