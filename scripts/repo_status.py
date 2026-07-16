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

from tt_rqm_kernels.benchmark_release import validate_release as validate_qmul_release
from tt_rqm_kernels.entanglement_benchmark import validate_entanglement_preregistration
from tt_rqm_kernels.hamiltonian_lowering_preregistration import (
    load_preregistration as load_h2a_preregistration,
)
from tt_rqm_kernels.hamiltonian_lowering_release import (
    HamiltonianLoweringReleaseError,
    validate_release as validate_h2a_release,
)
from tt_rqm_kernels.hamiltonian_lowering_pilot import (
    HamiltonianLoweringPilotError,
    validate_pilot_package as validate_h2a_pilot,
)
from tt_rqm_kernels.hamiltonian_lowering_designated import (
    HamiltonianLoweringDesignatedError,
    MANIFEST_PATH as H2A_DESIGNATED_MANIFEST,
    validate_designated_manifest,
)
from tt_rqm_kernels.hamiltonian_lowering_evidence import (
    HamiltonianLoweringEvidenceError,
    validate_clean_reproduction,
)
from tt_rqm_kernels.su2_benchmark import validate_su2_preregistration
from tt_rqm_kernels.su2_benchmark_release import (
    published_manifest_path,
    validate_release as validate_su2_release,
)

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
    h2a_status, h2a_detail = _h2a_foundation_status(repo_root)
    h2b_status, h2b_detail = _h2b_foundation_status(repo_root)
    entanglement_foundation_status, entanglement_foundation_detail = (
        _entanglement_foundation_status(repo_root)
    )
    entanglement_hardware_status, entanglement_hardware_detail = _entanglement_hardware_status(
        repo_root
    )

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
                "SU2ComposeBench current release",
                su2_comparison_status,
                su2_comparison_detail,
            ),
            _item(
                "SU2ComposeBench stability",
                su2_stability_status,
                su2_stability_detail,
            ),
            _item(
                "HamiltonianLoweringBench H2A",
                h2a_status,
                h2a_detail,
            ),
            _item(
                "HamiltonianEvolutionBench H2B",
                h2b_status,
                h2b_detail,
            ),
            _item(
                "EntanglementDynamicsBench reference foundation",
                entanglement_foundation_status,
                entanglement_foundation_detail,
            ),
            _item(
                "EntanglementDynamicsBench hardware",
                entanglement_hardware_status,
                entanglement_hardware_detail,
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


def _h2a_foundation_status(repo_root: Path) -> tuple[str, str]:
    required = (
        repo_root / "tt_rqm_kernels" / "hamiltonian_lowering_benchmark.py",
        repo_root / "tt_rqm_kernels" / "hamiltonian_lowering_candidate.py",
        repo_root / "scripts" / "hamiltonian_lowering_external_reference.py",
        repo_root / "scripts" / "validate_hamiltonian_lowering_candidate.py",
        repo_root
        / "experimental"
        / "tt_metalium_hamiltonian_lowering"
        / "audit_pinned_tt_metal.py",
    )
    if not all(path.is_file() for path in required):
        return "not implemented", "The H2A reference/candidate foundation is incomplete."
    designated = repo_root / "benchmarks" / "manifests" / "wormhole-hamiltonian-lowering.json"
    if designated.exists():
        try:
            release = validate_h2a_release(designated, repo_root=repo_root)
        except (HamiltonianLoweringReleaseError, OSError, ValueError, TypeError):
            return (
                "invalid Claim Level 0 release",
                "The public H2A release failed its hash, provenance, qualification, or generated-output gate.",
            )
        claim = release["claim"]
        return (
            "Claim Level 0 silicon conformance present",
            f"One designated N300 device-0 session passed all nine frozen H2A cases. "
            f"stable_benchmark={str(claim['stable_benchmark']).lower()} and "
            f"performance_eligible={str(claim['performance_eligible']).lower()}; this is not a performance claim.",
        )
    try:
        manifest = load_h2a_preregistration(
            repo_root / "benchmarks" / "manifests" / "hamiltonian-lowering-h2a-preregistration.json"
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return "invalid reference foundation", "The H2A preregistration failed validation."
    if manifest.get("status") != "pre_hardware":
        return "invalid reference foundation", "H2A must remain explicitly pre-hardware."
    pilot_root = repo_root / "benchmarks" / "pilots" / "hamiltonian-lowering-h2a"
    pilot_manifests = sorted(pilot_root.glob("*/pilot-manifest.json"))
    if pilot_manifests:
        try:
            result = validate_h2a_pilot(pilot_manifests[-1].parent)
        except HamiltonianLoweringPilotError:
            return (
                "invalid non-designated hardware pilot",
                "The retained H2A pilot failed offline package validation.",
            )
        status = (
            "non-designated hardware pilot passed"
            if result["pilot_passed"]
            else "non-designated hardware pilot failed"
        )
        frozen = repo_root / H2A_DESIGNATED_MANIFEST
        if result["pilot_passed"] and frozen.is_file():
            try:
                validate_designated_manifest(frozen, repo_root)
                validate_clean_reproduction(repo_root)
            except (HamiltonianLoweringDesignatedError, HamiltonianLoweringEvidenceError):
                return (
                    "invalid frozen designated contract",
                    "The clean reproduction evidence or frozen H2A contract failed validation.",
                )
            return (
                "candidate frozen for designated collection",
                "The clean committed candidate was reproducibly rebuilt and revalidated; the Claim Level 0 contract is frozen, but designated collection has not started and claim_level remains null.",
            )
        return (
            status,
            "The retained pilot is non-designated and qualification-ineligible; official H2A silicon conformance remains pending.",
        )
    candidate = (
        repo_root
        / "experimental"
        / "tt_metalium_hamiltonian_lowering"
        / "src"
        / "hamiltonian_lowering_candidate.cpp"
    )
    compute = (
        repo_root
        / "experimental"
        / "tt_metalium_hamiltonian_lowering"
        / "src"
        / "kernels"
        / "compute_hamiltonian_lowering.cpp"
    )
    if candidate.is_file() and compute.is_file():
        blocker = pilot_root / "h2a-n300-development-blocker-20260716" / "blocker-report.json"
        detail = (
            "Real single-core TT-Metal candidate source is present. The retained N300 large-angle development blocker stopped collection before the nine-case pilot; designated conformance remains absent."
            if blocker.is_file()
            else "Real single-core TT-Metal candidate source is present; non-designated pilot and designated conformance are absent."
        )
        return "TT-Metal candidate source present", detail
    return (
        "implementation-ready reference foundation",
        "CPU reference, independent oracles, external candidate protocol, pinned-API design audit, and Claim Level 0 preregistration are present; hardware execution is not implemented.",
    )


def _h2b_foundation_status(repo_root: Path) -> tuple[str, str]:
    required = (
        repo_root / "tt_rqm_kernels/hamiltonian/su2_evolution.py",
        repo_root / "tt_rqm_kernels/hamiltonian_evolution_benchmark.py",
        repo_root / "tt_rqm_kernels/hamiltonian_evolution_candidate.py",
        repo_root / "scripts/hamiltonian_evolution_external_reference.py",
        repo_root / "scripts/validate_hamiltonian_evolution_candidate.py",
        repo_root
        / "experimental/tt_metalium_hamiltonian_evolution/src/hamiltonian_evolution_candidate.cpp",
        repo_root / "experimental/tt_metalium_hamiltonian_evolution/CMakeLists.txt",
    )
    if not all(path.is_file() for path in required):
        return "not implemented", "The H2B reference, protocol, or TT-Metal candidate is absent."
    source = required[-2].read_text(encoding="utf-8")
    invariants = (
        source.count("MeshDevice::create_unit_mesh(0)") == 1,
        source.count("EnqueueMeshWorkload") == 2,
        source.count("EnqueueWriteMeshBuffer") == 1,
        source.count("EnqueueReadMeshBuffer") == 1,
        "build_h2a_program(device, input, intermediate" in source,
        "build_h1_program(device, intermediate, final_output" in source,
        '"program_count", 2' in source,
        '"host_round_trip_count", 0' in source,
    )
    if not all(invariants):
        return "invalid candidate architecture", "The H2B device-resident source audit failed."
    qualification_path = (
        repo_root / "benchmarks/processed/hamiltonian-evolution-h2b-pilot-qualification.json"
    )
    if qualification_path.is_file():
        try:
            qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return "invalid pilot qualification", "The H2B processed pilot result is malformed."
        if qualification.get("package_valid") is not True:
            return "invalid pilot qualification", "The retained H2B pilot package is invalid."
        if qualification.get("pilot_passed") is False:
            return (
                "first non-designated N300 pilot retained; did not pass (environment)",
                "All 20 frozen cases were attempted once without retry or replacement and stopped at TT-Metal runtime-root initialization before device execution. No H2B hardware claim exists.",
            )
    return (
        "CPU/reference foundation implemented; TT-Metal candidate source present; hardware not yet run",
        "Two programs share one Wormhole device session and a device-DRAM intermediate; stable_benchmark=false, performance_eligible=false, and claim_level=null.",
    )


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
    release_path = REPO_ROOT / "benchmarks/manifests/wormhole-qmul-level2.json"
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
        not release_path.exists()
        or not conformance_path.exists()
        or not performance_path.exists()
        or not all(path.exists() for path in companions)
    ):
        return "not implemented", "The protected persistent-device evidence set is incomplete."
    try:
        conformance = json.loads(conformance_path.read_text(encoding="utf-8"))
        performance = json.loads(performance_path.read_text(encoding="utf-8"))
        release = validate_qmul_release(
            release_path,
            repo_root=REPO_ROOT,
            verify_generated=False,
        )
        results = performance.get("results", [])
        valid = (
            release.get("claim")
            == {
                "level": 2,
                "name": "stable_one_device_performance",
                "public_session_count": 3,
                "stable_benchmark": True,
            }
            and conformance.get("measurement_mode") == "persistent_device_session.v1"
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
    except (json.JSONDecodeError, OSError, TypeError, AttributeError, ValueError):
        valid = False
    if not valid:
        return (
            "invalid persistent hardware report",
            "The persistent evidence failed lifecycle or report checks.",
        )
    return (
        "stable one-device performance present",
        "Three qualified device-0 sessions support aggregate Claim Level 2 with stable_benchmark=true; each source session remains false and this is not an acceleration claim.",
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
    manifest_path = root / published_manifest_path(root)
    if not manifest_path.is_file():
        return "not implemented", "The published SU2ComposeBench release is absent."
    try:
        release = validate_su2_release(manifest_path, repo_root=root, verify_generated=False)
        claim = release.get("claim")
        if claim not in (
            {
                "level": 1,
                "name": "qualified_first_comparison_sample",
                "public_session_count": 1,
                "stable_benchmark": False,
            },
            {
                "level": 2,
                "name": "stable_one_device_performance",
                "public_session_count": 3,
                "stable_benchmark": True,
            },
        ):
            raise ValueError("comparison claim or session count mismatch")
        report = json.loads((root / release["primary_report"]).read_text(encoding="utf-8"))
        if not _valid_su2_report_header(report, stage="performance", eligible=True):
            raise ValueError("comparison report header or device scope mismatch")
        fused_only = report.get("benchmark_mode") == "fused_stability"
        if not all(
            _valid_su2_result(result, samples=10, eligible=True, fused_only=fused_only)
            for result in report["results"]
        ):
            raise ValueError("comparison result validation failed")
    except (KeyError, OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return (
            "invalid comparison evidence",
            f"Published SU2 release validation failed: {exc}",
        )
    if claim["level"] == 2:
        return (
            "stable one-device fused performance present",
            "Three designated device-0 sessions passed the hash-bound SU2 Level 2 fused stability qualification; this is not an acceleration result.",
        )
    return (
        "qualified first comparison present",
        "One hash-bound fused/unfused device-0 session passed Claim Level 1 gates with stable_benchmark=false; it is not an acceleration result.",
    )


def _su2_stability_status(root: Path) -> tuple[str, str]:
    level2_path = root / "benchmarks/manifests/wormhole-su2-compose-level2.json"
    if level2_path.is_file():
        try:
            release = validate_su2_release(
                level2_path,
                repo_root=root,
                verify_generated=False,
            )
            if release.get("claim") != {
                "level": 2,
                "name": "stable_one_device_performance",
                "public_session_count": 3,
                "stable_benchmark": True,
            }:
                raise ValueError("Level 2 claim shape mismatch")
        except (OSError, TypeError, ValueError) as exc:
            return "invalid stability status", f"SU2 stability validation failed: {exc}"
        return (
            "established",
            "Three designated cold-start sessions passed the reproducible SU2 stability qualification.",
        )
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
    qualification_path = (
        root / "benchmarks/processed/wormhole-su2-compose-stability-qualification.json"
    )
    if qualification_path.is_file():
        try:
            qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
            if (
                qualification.get("qualification_passed") is not False
                or qualification.get("stable_benchmark") is not False
                or len(qualification.get("session_ids", [])) != 3
            ):
                raise ValueError("retained non-qualifying campaign shape mismatch")
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            return "invalid stability status", f"SU2 campaign validation failed: {exc}"
        return (
            "not established",
            "Three designated v2 sessions were retained, but the deterministic qualifier rejected frozen variability gates.",
        )
    return (
        "not established",
        "Only one public cold-start comparison session exists; Level 2 requires three qualified independent sessions.",
    )


def _entanglement_foundation_status(root: Path) -> tuple[str, str]:
    preregistration = root / "benchmarks/manifests/entanglement-dynamics-preregistration.json"
    required_sources = (
        root / "tt_rqm_kernels/hamiltonian/two_qubit.py",
        root / "tt_rqm_kernels/hamiltonian/two_qubit_metrics.py",
        root / "tt_rqm_kernels/hamiltonian/__init__.py",
    )
    if not preregistration.is_file() or not all(path.is_file() for path in required_sources):
        return "not implemented", "The H3 CPU reference package or preregistration is absent."
    try:
        payload = json.loads(preregistration.read_text(encoding="utf-8"))
        validate_entanglement_preregistration(payload)
        public_api = required_sources[-1].read_text(encoding="utf-8")
        required_api = {
            "lower_two_qubit_hamiltonian",
            "compose_two_qubit_state",
            "evolve_two_qubit_state_reference",
            "apply_local_rotor_pair",
            "two_qubit_state_diagnostics",
            "compare_two_qubit_states",
        }
        if not all(name in public_api for name in required_api):
            raise ValueError("public H3 reference API is incomplete")
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return "invalid reference foundation", f"H3 reference validation failed: {exc}"
    return (
        "implemented reference",
        "CPU two-qubit lowering, ordered state evolution, complex128 oracle, and typed entanglement diagnostics are present; no hardware evidence or claim level exists.",
    )


def _entanglement_hardware_status(root: Path) -> tuple[str, str]:
    forbidden_release = root / "benchmarks/manifests/wormhole-entanglement-dynamics.json"
    forbidden_report = root / "reports/tt_hardware_entanglement_dynamics.json"
    if forbidden_release.exists() or forbidden_report.exists():
        return (
            "unexpected evidence present",
            "H3 foundation scope forbids a hardware release or report.",
        )
    return (
        "not implemented",
        "EntanglementDynamicsBench is CPU-reference-only; no TT-Metalium path, hardware evidence, or claim level exists.",
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


def _valid_su2_result(
    result: dict[str, Any], *, samples: int, eligible: bool, fused_only: bool = False
) -> bool:
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
    paths = ("fused",) if fused_only else ("fused", "unfused")
    for path in paths:
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
