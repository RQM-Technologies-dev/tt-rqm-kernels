#!/usr/bin/env python3
"""Placeholder external-qmul command for a future TT-Metalium qmul candidate.

This file intentionally does not implement qmul. It exists so the candidate
package has a concrete command boundary that fails safely until real
TT-Metalium host/kernel source is added in an SDK environment.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experimental.tt_metalium_qmul.check_environment import _root_from_env


PROTOCOL = "tt-rqm-external-qmul.v1"


def main() -> int:
    work_dir = os.environ.get("TT_RQM_EXTERNAL_QMUL_DIR")
    manifest_path = os.environ.get("TT_RQM_EXTERNAL_QMUL_MANIFEST")
    if not work_dir or not manifest_path:
        print(
            "external-qmul environment missing: TT_RQM_EXTERNAL_QMUL_DIR and "
            "TT_RQM_EXTERNAL_QMUL_MANIFEST are required.",
            file=sys.stderr,
        )
        return 2

    manifest = _load_manifest(Path(manifest_path))
    if manifest.get("schema") != PROTOCOL:
        print(f"unsupported external-qmul protocol: {manifest.get('schema')!r}", file=sys.stderr)
        return 2
    if manifest.get("workload") != "qmul":
        print(f"unsupported workload: {manifest.get('workload')!r}", file=sys.stderr)
        return 2
    if manifest.get("dtype") != "float32":
        print(f"unsupported dtype: {manifest.get('dtype')!r}", file=sys.stderr)
        return 2

    root = _root_from_env()
    if root is None:
        print(
            "TT-Metalium SDK unavailable: this placeholder wrote no output. "
            "Set TT_METAL_HOME or TT_METALIUM_HOME to a real tt-metal checkout "
            "before replacing this placeholder with a real candidate command.",
            file=sys.stderr,
        )
        return 2

    print(
        "TT-Metalium SDK root was provided, but no TT-Metalium qmul "
        "implementation is present in this scaffold yet. No out.bin or "
        "metrics.json was written; no hardware performance is claimed.",
        file=sys.stderr,
    )
    return 2


def _load_manifest(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"external-qmul manifest not found: {path}", file=sys.stderr)
        raise SystemExit(2) from None
    if not isinstance(payload, dict):
        print("external-qmul manifest must contain a JSON object.", file=sys.stderr)
        raise SystemExit(2)
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
