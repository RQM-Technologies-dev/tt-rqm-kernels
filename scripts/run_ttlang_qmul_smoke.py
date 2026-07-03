#!/usr/bin/env python3
"""Run the optional TT-Lang simulator qmul smoke benchmark."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tt_rqm_kernels.backends.tt_lang import SETUP_HINT, check_tt_lang_sim

KERNEL_SCRIPT = REPO_ROOT / "tt_rqm_kernels" / "backends" / "tt_lang" / "qmul_sim_kernel.py"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run TT-Lang simulator qmul and emit a StructuredBench report."
    )
    parser.add_argument("--items", type=_positive_int, default=128)
    parser.add_argument("--iters", type=_positive_int, default=1)
    parser.add_argument("--warmup", type=_nonnegative_int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-output", type=Path, default=Path("reports/tt_lang_qmul_sim.json"))
    parser.add_argument("--markdown-output", type=Path, default=Path("reports/tt_lang_qmul_sim.md"))
    parser.add_argument("--sim-cli", default=None)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check simulator availability without running the benchmark.",
    )
    args = parser.parse_args(argv)

    availability = check_tt_lang_sim(sim_cli=args.sim_cli)
    if args.check:
        print(
            json.dumps(
                {
                    "available": availability.available,
                    "sim_cli": availability.sim_cli,
                    "version": availability.version,
                    "reason": availability.reason,
                    "setup_hint": availability.setup_hint,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if not availability.available or availability.sim_cli is None:
        print(availability.reason, file=sys.stderr)
        print(SETUP_HINT, file=sys.stderr)
        return 2

    env = os.environ.copy()
    env["PYTHONPATH"] = _pythonpath_with_repo(env.get("PYTHONPATH"))
    command = [
        availability.sim_cli,
        str(KERNEL_SCRIPT),
        "--items",
        str(args.items),
        "--iters",
        str(args.iters),
        "--warmup",
        str(args.warmup),
        "--seed",
        str(args.seed),
        "--json-output",
        str(args.json_output),
        "--markdown-output",
        str(args.markdown_output),
    ]
    completed = subprocess.run(command, text=True, env=env)
    return completed.returncode


def _pythonpath_with_repo(existing: str | None) -> str:
    repo = str(REPO_ROOT)
    if not existing:
        return repo
    parts = existing.split(os.pathsep)
    if repo in parts:
        return existing
    return os.pathsep.join([repo, existing])


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
