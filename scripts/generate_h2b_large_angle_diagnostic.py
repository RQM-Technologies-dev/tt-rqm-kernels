#!/usr/bin/env python3
"""Generate the deterministic H2B large-angle development diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tt_rqm_kernels.hamiltonian_evolution_diagnostics import (
    build_large_angle_diagnostic,
    render_large_angle_diagnostic,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json-output", type=Path, default=ROOT / "reports/h2b_large_angle_diagnostic.json"
    )
    parser.add_argument(
        "--markdown-output", type=Path, default=ROOT / "reports/h2b_large_angle_diagnostic.md"
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    report = build_large_angle_diagnostic(ROOT)
    rendered_json = json.dumps(report, indent=2, sort_keys=True) + "\n"
    rendered_markdown = render_large_angle_diagnostic(report)
    if args.check:
        if args.json_output.read_text(encoding="utf-8") != rendered_json:
            print("H2B large-angle JSON diagnostic is stale")
            return 1
        if args.markdown_output.read_text(encoding="utf-8") != rendered_markdown:
            print("H2B large-angle Markdown diagnostic is stale")
            return 1
    else:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered_json, encoding="utf-8")
        args.markdown_output.write_text(rendered_markdown, encoding="utf-8")
    print(
        "H2B large-angle diagnostic valid: "
        f"acceptance={report['diagnosis']['acceptance_path']} "
        f"sweep_cases={report['sweep']['case_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
