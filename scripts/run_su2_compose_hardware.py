#!/usr/bin/env python3
"""Collect a conformance or performance SU2ComposeBench hardware report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tt_rqm_kernels.backends.tenstorrent.su2_compose_persistent import render_su2_markdown, run_su2_compose


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", required=True)
    parser.add_argument("--stage", choices=("conformance", "performance"), required=True)
    parser.add_argument("--methodology-note", required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()
    report = run_su2_compose(command=args.command, stage=args.stage, methodology_note=args.methodology_note)
    args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.markdown_output.write_text(render_su2_markdown(report))
    print(args.json_output)
    print(args.markdown_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
