#!/usr/bin/env python3
"""Collect the frozen one-pass non-designated H2B N300 pilot."""

from __future__ import annotations

import argparse
from pathlib import Path

from tt_rqm_kernels.hamiltonian_evolution_pilot import collect_pilot

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", required=True)
    parser.add_argument("--preflight-command", required=True)
    parser.add_argument("--health-command", required=True)
    parser.add_argument("--environment-command", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pilot-id", required=True)
    args = parser.parse_args()
    suite = collect_pilot(
        repo_root=ROOT,
        output_dir=args.output_dir,
        pilot_id=args.pilot_id,
        command=args.command,
        preflight_command=args.preflight_command,
        health_command=args.health_command,
        environment_command=args.environment_command,
    )
    print(f"pilot_id={suite['pilot_id']}")
    print(f"suite_passed={str(suite['suite_passed']).lower()}")
    return 0 if suite["suite_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
