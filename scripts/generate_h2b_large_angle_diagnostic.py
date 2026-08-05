#!/usr/bin/env python3
"""Generate the deterministic H2B large-angle development diagnostic."""

from __future__ import annotations

import argparse
import difflib
import json
import os
import platform
import sys
from pathlib import Path

os.environ["ATEN_CPU_CAPABILITY"] = "default"
import torch

from tt_rqm_kernels.hamiltonian_evolution_diagnostics import (
    build_large_angle_diagnostic,
    DIAGNOSTIC_CPU_CAPABILITY,
    DIAGNOSTIC_PLATFORM,
    DIAGNOSTIC_TORCH_VERSION,
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
    generator_platform = f"{platform.system()}-{platform.machine()}"
    if generator_platform != DIAGNOSTIC_PLATFORM:
        print(
            "H2B large-angle diagnostic requires "
            f"{DIAGNOSTIC_PLATFORM}; found {generator_platform}.",
            file=sys.stderr,
        )
        return 2
    torch_version = torch.__version__.split("+", maxsplit=1)[0]
    if torch_version != DIAGNOSTIC_TORCH_VERSION:
        print(
            "H2B large-angle diagnostic requires "
            f"torch=={DIAGNOSTIC_TORCH_VERSION}; found torch=={torch_version}. "
            'Install the pinned environment with `python -m pip install -e ".[diagnostic]"`.',
            file=sys.stderr,
        )
        return 2
    if os.environ["ATEN_CPU_CAPABILITY"] != DIAGNOSTIC_CPU_CAPABILITY:
        print("H2B large-angle diagnostic CPU capability contract is not active", file=sys.stderr)
        return 2
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    report = build_large_angle_diagnostic(ROOT)
    rendered_json = json.dumps(report, indent=2, sort_keys=True) + "\n"
    rendered_markdown = render_large_angle_diagnostic(report)
    if args.check:
        committed_json = args.json_output.read_text(encoding="utf-8")
        if committed_json != rendered_json:
            print("H2B large-angle JSON diagnostic is stale")
            print(
                "".join(
                    difflib.unified_diff(
                        committed_json.splitlines(keepends=True),
                        rendered_json.splitlines(keepends=True),
                        fromfile=str(args.json_output),
                        tofile="regenerated JSON",
                    )
                ),
                end="",
            )
            return 1
        committed_markdown = args.markdown_output.read_text(encoding="utf-8")
        if committed_markdown != rendered_markdown:
            print("H2B large-angle Markdown diagnostic is stale")
            print(
                "".join(
                    difflib.unified_diff(
                        committed_markdown.splitlines(keepends=True),
                        rendered_markdown.splitlines(keepends=True),
                        fromfile=str(args.markdown_output),
                        tofile="regenerated Markdown",
                    )
                ),
                end="",
            )
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
