#!/usr/bin/env python3
"""Collect matched CPU evidence or validate a configured real N300 runner."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path

from tt_rqm_kernels.qsp_benchmark import (
    collect_cpu_report,
    validate_hardware_report,
    write_json_empty,
)

HARDWARE_COMMAND_ENV = "TT_RQM_QSP_HARDWARE_COMMAND"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("cpu", "conformance", "performance"), required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--session-id", default="qsp-cpu-development")
    parser.add_argument("--command")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[256, 4096])
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    args = parser.parse_args()
    if args.stage == "cpu":
        report = collect_cpu_report(
            sizes=tuple(args.sizes),
            warmups=args.warmups,
            repetitions=args.repetitions,
        )
        report["session_id"] = args.session_id
        write_json_empty(args.output, report)
        print(json.dumps(report, sort_keys=True, indent=2))
        return 0
    command = args.command or os.environ.get(HARDWARE_COMMAND_ENV)
    if not command:
        print(
            f"real N300 runner unavailable: set {HARDWARE_COMMAND_ENV} or pass --command; "
            "emulation, Docker, Python references, and historical qmul evidence are rejected"
        )
        return 2
    tokens = shlex.split(command)
    if not tokens or any(word in command.lower() for word in ("docker", "emule", "simulator")):
        print("configured QSP hardware command is empty or names a prohibited nonhardware path")
        return 2
    with tempfile.TemporaryDirectory(prefix="qsp-n300-") as temp:
        raw_path = Path(temp) / "hardware-report.json"
        completed = subprocess.run(
            [
                *tokens,
                "--protocol",
                "tt-rqm-qsp-kernel-benchmark.v1",
                "--stage",
                args.stage,
                "--device",
                str(args.device),
                "--session-id",
                args.session_id,
                "--output",
                str(raw_path),
            ],
            check=False,
        )
        if completed.returncode != 0 or not raw_path.is_file():
            print("configured real-hardware runner failed without an accepted report")
            return 2
        report = json.loads(raw_path.read_text())
    validate_hardware_report(report, expected_stage=args.stage)
    write_json_empty(args.output, report)
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
