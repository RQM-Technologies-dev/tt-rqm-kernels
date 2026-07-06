#!/usr/bin/env python3
"""Run the experimental TT-Metalium qmul candidate binary.

The binary is built by build_candidate.py when a real TT-Metalium package is
available. This wrapper preserves the external-qmul command boundary and fails
cleanly when the binary has not been built yet.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experimental.tt_metalium_qmul.check_environment import _root_from_env


PROTOCOL = "tt-rqm-external-qmul.v1"
DEFAULT_BINARY = (
    Path(__file__).resolve().parent
    / "build_emule_candidate"
    / "tt_rqm_metalium_qmul_candidate"
)


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
            "TT-Metalium SDK unavailable: the candidate wrote no output. "
            "Set TT_METAL_HOME or TT_METALIUM_HOME to a real tt-metal checkout "
            "before building the experimental candidate.",
            file=sys.stderr,
        )
        return 2

    binary = Path(os.environ.get("TT_RQM_METALIUM_QMUL_BINARY", DEFAULT_BINARY))
    if not binary.exists():
        print(
            "TT-Metalium qmul candidate binary not found: "
            f"{binary}. Run experimental/tt_metalium_qmul/build_candidate.py "
            "in a built TT-Metalium environment first. No out.bin or "
            "metrics.json was written.",
            file=sys.stderr,
        )
        return 2

    completed = subprocess.run([str(binary)], check=False)
    return completed.returncode


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
