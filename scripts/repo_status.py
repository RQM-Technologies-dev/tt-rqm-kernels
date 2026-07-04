"""Print a concise current-status report for tt-rqm-kernels."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
TT_LANG_AVAILABILITY_PATH = (
    REPO_ROOT / "tt_rqm_kernels" / "backends" / "tt_lang" / "availability.py"
)
_spec = importlib.util.spec_from_file_location(
    "_tt_rqm_tt_lang_availability",
    TT_LANG_AVAILABILITY_PATH,
)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"cannot load {TT_LANG_AVAILABILITY_PATH}")
_tt_lang_availability = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _tt_lang_availability
_spec.loader.exec_module(_tt_lang_availability)
check_tt_lang_sim = _tt_lang_availability.check_tt_lang_sim


def build_status() -> dict[str, Any]:
    tt_lang = check_tt_lang_sim()
    tt_metalium_dir = REPO_ROOT / "experimental" / "tt_metalium_qmul"
    tt_emule_dir = REPO_ROOT / "experimental" / "tt_emule_qmul"

    return {
        "schema": "tt-rqm-repo-status.v1",
        "platform": platform.platform(),
        "items": [
            _item(
                "CPU/PyTorch reference",
                "implemented",
                "tt_rqm_kernels quaternion, rotor, and phase reference kernels are present.",
            ),
            _item(
                "StructuredBench smoke",
                "implemented",
                "Run: python -m tt_rqm_kernels.structuredbench --suite smoke",
            ),
            _item(
                "TT-Lang simulator",
                "available" if tt_lang.available else "optional / unavailable",
                tt_lang.reason,
            ),
            _item(
                "external-qmul harness",
                "implemented",
                "Run: python scripts/validate_qmul_candidate.py --command \"python scripts/qmul_external_reference.py\"",
            ),
            _item(
                "TT-Metalium candidate",
                "source candidate present / not built",
                "experimental/tt_metalium_qmul contains a scalar RISC-V qmul candidate; no build/run report exists yet.",
            ),
            _item(
                "TT-Metalium scaffold",
                "implemented" if tt_metalium_dir.exists() else "missing",
                "experimental/tt_metalium_qmul contains source, preflight, build, run, and validation wrappers.",
            ),
            _item(
                "tt-emule preflight",
                "implemented" if (tt_emule_dir / "check_environment.py").exists() else "missing",
                "Run: python experimental/tt_emule_qmul/check_environment.py",
            ),
            _item(
                "tt-emule candidate",
                "not implemented",
                "Issue #8 remains open until the TT-Metalium candidate builds and runs under tt-emule.",
            ),
            _item(
                "hardware report",
                "not implemented",
                "Only the future report template exists; no hardware result is claimed.",
            ),
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print a concise implementation-status report for tt-rqm-kernels."
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON status.")
    args = parser.parse_args()

    status = build_status()
    if args.json:
        print(json.dumps(status, indent=2, sort_keys=True))
    else:
        print("tt-rqm-kernels current status")
        print(f"schema: {status['schema']}")
        print(f"platform: {status['platform']}")
        print("")
        for item in status["items"]:
            print(f"{item['name']}: {item['status']}")
            print(f"  {item['detail']}")
    return 0


def _item(name: str, status: str, detail: str) -> dict[str, str]:
    return {
        "name": name,
        "status": status,
        "detail": detail,
    }


if __name__ == "__main__":
    raise SystemExit(main())
