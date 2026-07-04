#!/usr/bin/env python3
"""One-command Tenstorrent qmul quickstart for tt-rqm-kernels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from tt_rqm_kernels.backends.tenstorrent.availability import (
    DEFAULT_HARDWARE_COMMAND_ENV,
    check_readiness,
    resolve_execution_path,
)
from tt_rqm_kernels.backends.tenstorrent.qmul_external import (
    TenstorrentAdapterError,
    run_configured_qmul,
)
from tt_rqm_kernels.backends.tenstorrent.report import (
    ReportLabelError,
    write_structuredbench_report,
)
from tt_rqm_kernels.structuredbench import render_table


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check or run the Tenstorrent external-qmul path. This is not a "
            "TT-NN integration and never labels emulation as hardware."
        )
    )
    parser.add_argument("--check", action="store_true", help="Print readiness checks.")
    parser.add_argument("--mode", choices=("emule", "hardware"), default=None)
    parser.add_argument(
        "--command",
        default=None,
        help=(
            "External qmul command. Hardware mode also reads "
            f"{DEFAULT_HARDWARE_COMMAND_ENV}."
        ),
    )
    parser.add_argument("--items", type=_positive_int, default=128)
    parser.add_argument("--iters", type=_positive_int, default=1)
    parser.add_argument("--warmup", type=_nonnegative_int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="StructuredBench JSON output path. Defaults by mode.",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=None,
        help="StructuredBench Markdown output path. Defaults by mode.",
    )
    parser.add_argument(
        "--stable-benchmark",
        action="store_true",
        help="Mark a real hardware run as stable only if methodology supports it.",
    )
    parser.add_argument("--format", choices=("table", "json"), default="table")
    args = parser.parse_args()

    if args.check:
        readiness = check_readiness(hardware_command=args.command)
        _print_readiness(readiness)
        return 0

    if args.mode is None:
        parser.error("--mode is required unless --check is set")

    path = resolve_execution_path(args.mode, command=args.command)
    if not path.available:
        print(path.reason, file=sys.stderr)
        if args.mode == "hardware":
            print(
                f"Set {DEFAULT_HARDWARE_COMMAND_ENV} or pass --command for a real "
                "Tenstorrent Cloud/hardware candidate. Do not use the tt-emule "
                "wrapper for hardware-labeled reports.",
                file=sys.stderr,
            )
        return 2

    json_output, markdown_output = _default_outputs(
        args.mode,
        json_output=args.json_output,
        markdown_output=args.markdown_output,
    )

    try:
        report = run_configured_qmul(
            args.mode,
            command=path.command,
            items=args.items,
            iterations=args.iters,
            warmup=args.warmup,
            seed=args.seed,
            stable_benchmark=args.stable_benchmark,
        )
    except (TenstorrentAdapterError, ReportLabelError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    write_structuredbench_report(
        report,
        json_output=json_output,
        markdown_output=markdown_output,
    )
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_table(report))
        print(f"JSON report: {json_output}")
        print(f"Markdown report: {markdown_output}")
    return 0


def _print_readiness(readiness) -> None:
    print("RQM Tenstorrent qmul quickstart readiness")
    print(f"repo root: {readiness.repo_root}")
    print(f"report output directory: {readiness.report_dir}")
    print("")
    for item in readiness.checks:
        status = "ok" if item.available else "missing"
        suffix = f" ({item.path})" if item.path else ""
        print(f"{item.name}: {status} - {item.detail}{suffix}")
    print("")
    print(f"emule mode ready: {str(readiness.emule_ready).lower()}")
    print(f"hardware mode ready: {str(readiness.hardware_ready).lower()}")
    if not readiness.hardware_ready:
        print(
            f"hardware command: set {DEFAULT_HARDWARE_COMMAND_ENV} or pass --command"
        )


def _default_outputs(
    mode: str,
    *,
    json_output: Path | None,
    markdown_output: Path | None,
) -> tuple[Path, Path]:
    stem = "tt_emule_qmul_quickstart" if mode == "emule" else "tt_hardware_qmul_quickstart"
    return (
        json_output or Path("reports") / f"{stem}.json",
        markdown_output or Path("reports") / f"{stem}.md",
    )


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
