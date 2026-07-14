"""Print a concise current-status report for tt-rqm-kernels."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tt_rqm_kernels.su2_benchmark import validate_su2_preregistration
from tt_rqm_kernels.su2_benchmark_release import validate_release as validate_su2_release

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


def build_status(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    tt_lang = check_tt_lang_sim()
    tt_metalium_dir = REPO_ROOT / "experimental" / "tt_metalium_qmul"
    tt_emule_dir = REPO_ROOT / "experimental" / "tt_emule_qmul"
    tt_emule_report = REPO_ROOT / "reports" / "tt_emule_qmul_candidate.json"
    hardware_report_status, hardware_report_detail = _hardware_report_status()
    stage_b_status, stage_b_detail = _stage_b_report_status()
    persistent_status, persistent_detail = _persistent_stage_b_report_status()
    su2_foundation_status, su2_foundation_detail = _su2_foundation_status(repo_root)
    su2_conformance_status, su2_conformance_detail = _su2_conformance_status(repo_root)
    su2_comparison_status, su2_comparison_detail = _su2_comparison_status(repo_root)
    su2_stability_status, su2_stability_detail = _su2_stability_status(repo_root)

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
                'Run: python scripts/validate_qmul_candidate.py --command "python scripts/qmul_external_reference.py"',
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
            _item(
                "Persistent Stage B hardware report",
                persistent_status,
                persistent_detail,
            ),
            _item(
                "SU2ComposeBench reference foundation",
                su2_foundation_status,
                su2_foundation_detail,
            ),
            _item(
                "SU2ComposeBench N300 conformance",
                su2_conformance_status,
                su2_conformance_detail,
            ),
            _item(
                "SU2ComposeBench first comparison",
                su2_comparison_status,
                su2_comparison_detail,
            ),
            _item(
                "SU2ComposeBench stability",
                su2_stability_status,
                su2_stability_detail,
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
    conformance_path = REPO_ROOT / "reports" / "tt_hardware_qmul_stage_b_candidate_conformance.json"
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


def _persistent_stage_b_report_status() -> tuple[str, str]:
    conformance_path = (
        REPO_ROOT / "reports" / "tt_hardware_qmul_stage_b_persistent_conformance.json"
    )
    performance_path = (
        REPO_ROOT / "reports" / "tt_hardware_qmul_stage_b_persistent_performance.json"
    )
    companions = (
        REPO_ROOT / "reports" / "tt_hardware_qmul_stage_b_persistent_conformance.md",
        REPO_ROOT / "reports" / "tt_hardware_qmul_stage_b_persistent_performance.md",
        REPO_ROOT / "reports" / "tt_hardware_qmul_stage_b_persistent_environment.txt",
        REPO_ROOT / "reports" / "tt_hardware_qmul_stage_b_persistent_timing_audit.md",
    )
    if (
        not conformance_path.exists()
        or not performance_path.exists()
        or not all(path.exists() for path in companions)
    ):
        return "not implemented", "The protected persistent-device evidence set is incomplete."
    try:
        conformance = json.loads(conformance_path.read_text(encoding="utf-8"))
        performance = json.loads(performance_path.read_text(encoding="utf-8"))
        results = performance.get("results", [])
        valid = (
            conformance.get("measurement_mode") == "persistent_device_session.v1"
            and conformance.get("stable_benchmark") is False
            and conformance.get("lifecycle", {}).get("create_count") == 1
            and conformance.get("lifecycle", {}).get("close_count") == 1
            and performance.get("benchmark_stage") == "performance"
            and performance.get("case_items") == [4096, 65536, 262144]
            and performance.get("stable_benchmark") is False
            and performance.get("lifecycle")
            == {"close_count": 1, "create_count": 1, "device_count": 1, "device_id": 0}
            and len(results) == 3
            and all(
                result.get("correctness", {}).get("passed") is True
                and result.get("implementation_class") == "multicore_tensix_sfpu_qmul_persistent"
                and result.get("performance_eligible") is True
                and result.get("timing", {}).get("repetitions") == 10
                for result in results
            )
        )
    except (json.JSONDecodeError, OSError, TypeError, AttributeError):
        valid = False
    if not valid:
        return (
            "invalid persistent hardware report",
            "The persistent evidence failed lifecycle or report checks.",
        )
    return (
        "first persistent hardware sample present",
        "One device-0 lifecycle completed the full sweep with whole-output validation and stable_benchmark=false; it is not an acceleration claim.",
    )


def _su2_foundation_status(root: Path) -> tuple[str, str]:
    preregistration = root / "benchmarks/manifests/su2-compose-preregistration.json"
    required_sources = (
        root / "tt_rqm_kernels/hamiltonian/__init__.py",
        root / "tt_rqm_kernels/hamiltonian/su2_lowering.py",
        root / "tt_rqm_kernels/hamiltonian/su2_compose.py",
        root / "tt_rqm_kernels/hamiltonian/su2_reference.py",
    )
    if not preregistration.is_file() or not all(path.is_file() for path in required_sources):
        return "not implemented", "The H1 reference package or preregistration is absent."
    try:
        payload = json.loads(preregistration.read_text(encoding="utf-8"))
        validate_su2_preregistration(payload)
        public_api = (root / "tt_rqm_kernels/hamiltonian/__init__.py").read_text(encoding="utf-8")
        required_api = {
            "lower_two_level_hamiltonian",
            "su2_compose_chain",
            "u2_matrix_from_rotor_phase",
            "compose_hamiltonian_matrices",
        }
        if not all(name in public_api for name in required_api):
            raise ValueError("public H1 reference API is incomplete")
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return "invalid reference foundation", f"H1 reference validation failed: {exc}"
    return (
        "implemented reference",
        "CPU lowering, Float64 composition, complex128 matrix oracle, and preregistered H1 contract are present.",
    )


def _su2_conformance_status(root: Path) -> tuple[str, str]:
    manifest_path = root / "benchmarks/manifests/su2-compose-conformance.json"
    report_path = root / "reports/tt_hardware_su2_compose_conformance.json"
    eligible_path = root / "reports/tt_hardware_su2_compose_eligible_conformance.json"
    performance_manifest_path = root / "benchmarks/manifests/wormhole-su2-compose.json"
    if not all(
        path.is_file()
        for path in (manifest_path, report_path, eligible_path, performance_manifest_path)
    ):
        return "not implemented", "The hash-bound SU2ComposeBench conformance release is absent."
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != "tt-rqm-su2-compose-conformance-release.v1":
            raise ValueError("conformance release schema mismatch")
        if manifest.get("claim") != {
            "level": 0,
            "name": "silicon_conformance",
            "stable_benchmark": False,
        }:
            raise ValueError("conformance claim mismatch")
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, list) or {
            artifact.get("path") for artifact in artifacts if isinstance(artifact, dict)
        } != {
            "reports/tt_hardware_su2_compose_conformance.json",
            "reports/tt_hardware_su2_compose_conformance.md",
            "reports/tt_hardware_su2_compose_environment.txt",
            "reports/tt_hardware_su2_compose_architecture_audit.md",
        }:
            raise ValueError("conformance artifact set mismatch")
        for artifact in artifacts:
            artifact_path = root / artifact["path"]
            if not artifact_path.is_file():
                raise ValueError(f"missing conformance artifact: {artifact['path']}")
            observed = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            if observed != artifact.get("sha256"):
                raise ValueError(f"conformance artifact hash mismatch: {artifact['path']}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not _valid_su2_report_header(report, stage="conformance", eligible=False):
            raise ValueError("conformance report header or device scope mismatch")
        results = report.get("results")
        if not isinstance(results, list) or [(r.get("B"), r.get("K")) for r in results] != [
            (32, 8),
            (2048, 8),
        ]:
            raise ValueError("conformance case set mismatch")
        if not all(_valid_su2_result(result, samples=1, eligible=False) for result in results):
            raise ValueError("conformance result validation failed")
        candidate = report["provenance"]["candidate"]
        if candidate.get("candidate_sha256") != manifest["provenance"]["candidate_sha256"]:
            raise ValueError("conformance candidate identity mismatch")

        performance_manifest = json.loads(performance_manifest_path.read_text(encoding="utf-8"))
        eligible_artifact = next(
            (
                artifact
                for artifact in performance_manifest.get("artifacts", [])
                if artifact.get("path")
                == "reports/tt_hardware_su2_compose_eligible_conformance.json"
            ),
            None,
        )
        if eligible_artifact is None:
            raise ValueError("eligible conformance report is not hash-bound")
        observed_eligible_hash = hashlib.sha256(eligible_path.read_bytes()).hexdigest()
        if observed_eligible_hash != eligible_artifact.get("sha256"):
            raise ValueError("eligible conformance artifact hash mismatch")
        eligible_report = json.loads(eligible_path.read_text(encoding="utf-8"))
        if not _valid_su2_report_header(eligible_report, stage="conformance", eligible=True):
            raise ValueError("eligible conformance header or device scope mismatch")
        eligible_results = eligible_report.get("results")
        if not isinstance(eligible_results, list) or [
            (result.get("B"), result.get("K")) for result in eligible_results
        ] != [(32, 8), (2048, 8)]:
            raise ValueError("eligible conformance case set mismatch")
        if not all(
            _valid_su2_result(result, samples=1, eligible=True) for result in eligible_results
        ):
            raise ValueError("eligible conformance result validation failed")
        if (
            eligible_report["provenance"]["candidate"]["candidate_sha256"]
            != performance_manifest["provenance"]["candidate_sha256"]
        ):
            raise ValueError("eligible conformance candidate identity mismatch")
    except (KeyError, OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return "invalid conformance evidence", f"SU2 N300 conformance validation failed: {exc}"
    return (
        "hardware conformance present",
        "Fused and unfused H1 paths passed whole-output validation on Wormhole device 0; this is Claim Level 0.",
    )


def _su2_comparison_status(root: Path) -> tuple[str, str]:
    manifest_path = root / "benchmarks/manifests/wormhole-su2-compose.json"
    if not manifest_path.is_file():
        return "not implemented", "The SU2ComposeBench Claim Level 1 release is absent."
    try:
        release = validate_su2_release(manifest_path, repo_root=root, verify_generated=False)
        claim = release.get("claim")
        if claim != {
            "level": 1,
            "name": "qualified_first_comparison_sample",
            "public_session_count": 1,
            "stable_benchmark": False,
        }:
            raise ValueError("comparison claim or session count mismatch")
        report = json.loads((root / release["primary_report"]).read_text(encoding="utf-8"))
        if not _valid_su2_report_header(report, stage="performance", eligible=True):
            raise ValueError("comparison report header or device scope mismatch")
        if not all(
            _valid_su2_result(result, samples=10, eligible=True) for result in report["results"]
        ):
            raise ValueError("comparison result validation failed")
    except (KeyError, OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return "invalid comparison evidence", f"SU2 Claim Level 1 validation failed: {exc}"
    return (
        "qualified first comparison present",
        "One hash-bound fused/unfused device-0 session passed Claim Level 1 gates with stable_benchmark=false; it is not an acceleration result.",
    )


def _su2_stability_status(root: Path) -> tuple[str, str]:
    manifest_path = root / "benchmarks/manifests/wormhole-su2-compose.json"
    if not manifest_path.is_file():
        return (
            "not established",
            "No SU2 performance release exists; multi-session stability is absent.",
        )
    try:
        release = json.loads(manifest_path.read_text(encoding="utf-8"))
        claim = release.get("claim", {})
        sessions = release.get("sessions", [])
        if (
            claim.get("level") != 1
            or claim.get("stable_benchmark") is not False
            or claim.get("public_session_count") != 1
            or len(sessions) != 1
        ):
            raise ValueError("release is inconsistent with one non-stable Claim Level 1 session")
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return "invalid stability status", f"SU2 stability claim validation failed: {exc}"
    return (
        "not established",
        "Only one public cold-start comparison session exists; Level 2 requires three qualified independent sessions.",
    )


def _valid_su2_report_header(report: dict[str, Any], *, stage: str, eligible: bool) -> bool:
    return (
        report.get("schema") == "tt-rqm-su2-compose-report.v1"
        and report.get("benchmark") == "SU2ComposeBench"
        and report.get("family") == "SU2HamiltonianBench"
        and report.get("protocol") == "tt-rqm-external-su2-compose-persistent.v1"
        and report.get("execution_label") == "hardware"
        and report.get("benchmark_stage") == stage
        and report.get("performance_eligible") is eligible
        and report.get("stable_benchmark") is False
        and report.get("lifecycle")
        == {"close_count": 1, "create_count": 1, "device_count": 1, "device_id": 0}
    )


def _valid_su2_result(result: dict[str, Any], *, samples: int, eligible: bool) -> bool:
    batch = result.get("B")
    metadata = result.get("candidate_metadata", {})
    if not isinstance(batch, int) or batch <= 0:
        return False
    if (
        result.get("performance_eligible") is not eligible
        or result.get("stable_benchmark") is not False
        or result.get("samples") != samples
        or metadata.get("device_count") != 1
        or metadata.get("device_id") != 0
        or metadata.get("arithmetic_path") != "tensix_compute_sfpu"
        or metadata.get("fused_accumulator_storage") != "tensix_l1_ping_pong"
    ):
        return False
    for path in ("fused", "unfused"):
        correctness = result.get(path, {}).get("correctness", {})
        if (
            correctness.get("validated_values") != 6 * batch
            or correctness.get("failing_values") != 0
            or correctness.get("nonfinite_values") != 0
        ):
            return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
