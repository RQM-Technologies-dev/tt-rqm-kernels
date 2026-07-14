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
    tt_emule_report = REPO_ROOT / "reports" / "tt_emule_qmul_candidate.json"
    hardware_report_status, hardware_report_detail = _hardware_report_status()
    stage_b_status, stage_b_detail = _stage_b_report_status()

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
                "experimental source candidate present",
                "experimental/tt_metalium_qmul contains the immutable scalar Stage A baseline and separate multicore/SFPU Stage B candidate.",
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
                "emulation report present" if tt_emule_report.exists() else "not implemented",
                (
                    "reports/tt_emule_qmul_candidate.json is an emulation-labeled sample, not hardware performance."
                    if tt_emule_report.exists()
                    else "Issue #8 remains open until the TT-Metalium candidate builds and runs under tt-emule."
                ),
            ),
            _item(
                "hardware report",
                hardware_report_status,
                hardware_report_detail,
            ),
            _item(
                "Stage B hardware report",
                stage_b_status,
                stage_b_detail,
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


def _hardware_report_status() -> tuple[str, str]:
    report_path = REPO_ROOT / "reports" / "tt_hardware_qmul_quickstart.json"
    companion_paths = (
        REPO_ROOT / "reports" / "tt_hardware_qmul_quickstart.md",
        REPO_ROOT / "reports" / "tt_hardware_qmul_environment.txt",
    )
    if not report_path.exists() or not all(path.exists() for path in companion_paths):
        return (
            "not implemented",
            "The required JSON, Markdown, and environment evidence set is incomplete.",
        )
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        results = report.get("results", [])
        valid = (
            report.get("execution_label") == "hardware"
            and report.get("benchmark_stage") == "conformance"
            and report.get("stable_benchmark") is False
            and len(results) == 1
            and results[0].get("correctness", {}).get("passed") is True
            and results[0].get("performance_eligible") is False
        )
    except (json.JSONDecodeError, OSError, TypeError, AttributeError):
        valid = False
    if not valid:
        return (
            "invalid hardware report",
            "The committed hardware evidence does not satisfy the Stage A status checks.",
        )
    return (
        "hardware conformance report present",
        "reports/tt_hardware_qmul_quickstart.* records one N300 Stage A correctness run; it is not performance-eligible.",
    )


def _stage_b_report_status() -> tuple[str, str]:
    conformance_path = (
        REPO_ROOT / "reports" / "tt_hardware_qmul_stage_b_candidate_conformance.json"
    )
    performance_path = REPO_ROOT / "reports" / "tt_hardware_qmul_stage_b_performance.json"
    companion_paths = (
        REPO_ROOT / "reports" / "tt_hardware_qmul_stage_b_candidate_conformance.md",
        REPO_ROOT / "reports" / "tt_hardware_qmul_stage_b_performance.md",
        REPO_ROOT / "reports" / "tt_hardware_qmul_stage_b_architecture_audit.md",
    )
    if (
        not conformance_path.exists()
        or not performance_path.exists()
        or not all(path.exists() for path in companion_paths)
    ):
        return (
            "not implemented",
            "The protected Stage B conformance, audit, and performance evidence set is incomplete.",
        )
    try:
        conformance = json.loads(conformance_path.read_text(encoding="utf-8"))
        performance = json.loads(performance_path.read_text(encoding="utf-8"))
        conformance_results = conformance.get("results", [])
        performance_results = performance.get("results", [])
        valid = (
            conformance.get("execution_label") == "hardware"
            and conformance.get("benchmark_stage") == "conformance"
            and conformance.get("stable_benchmark") is False
            and len(conformance_results) == 1
            and conformance_results[0].get("correctness", {}).get("passed") is True
            and conformance_results[0].get("performance_eligible") is False
            and performance.get("execution_label") == "hardware"
            and performance.get("benchmark_stage") == "performance"
            and performance.get("stable_benchmark") is False
            and performance.get("repetitions") == 10
            and performance.get("case_items") == [4096, 65536, 262144]
            and len(performance_results) == 3
            and all(
                result.get("correctness", {}).get("passed") is True
                and result.get("performance_eligible") is True
                and result.get("candidate_metadata", {}).get("device_count") == 1
                and result.get("candidate_metadata", {}).get("device_id") == 0
                and result.get("candidate_metadata", {}).get("core_count", 0) > 1
                for result in performance_results
            )
        )
    except (json.JSONDecodeError, OSError, TypeError, AttributeError):
        valid = False
    if not valid:
        return (
            "invalid Stage B hardware report",
            "The committed Stage B evidence does not satisfy the protected methodology checks.",
        )
    return (
        "first hardware sample present",
        "The one-device multicore/SFPU sweep passed whole-output validation with performance_eligible=true and stable_benchmark=false; it is not an acceleration claim.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
