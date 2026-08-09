#!/usr/bin/env python3
"""Qualify exactly three independent QSP N300 performance sessions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tt_rqm_kernels.qsp_benchmark import qualify_hardware_sessions, write_json_empty


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", type=Path, nargs=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [json.loads(path.read_text()) for path in args.reports]
    result = qualify_hardware_sessions(reports)
    write_json_empty(args.output, result)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
