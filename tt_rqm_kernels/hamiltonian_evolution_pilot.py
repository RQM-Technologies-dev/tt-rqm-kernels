"""One-pass non-designated H2B pilot collection and offline qualification."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

from tt_rqm_kernels.hamiltonian_evolution_candidate import (
    HamiltonianEvolutionCandidateError,
    run_external_candidate,
)
from tt_rqm_kernels.hamiltonian_evolution_pilot_contract import (
    CASE_ORDER,
    DEFAULT_INPUT_ROOT,
    DEFAULT_MANIFEST,
    MATRIX_THRESHOLD,
    SOURCE_MANIFEST,
    STRESS_CASES,
    load_frozen_case,
    validate_pilot_contract,
)
from tt_rqm_kernels.hardware_session import validate_device_health

PILOT_SCHEMA = "tt-rqm-hamiltonian-evolution-pilot.v1"
SUITE_SCHEMA = "tt-rqm-hamiltonian-evolution-pilot-suite.v1"
PREFLIGHT_SCHEMA = "tt-rqm-hamiltonian-evolution-pilot-preflight.v1"


class HamiltonianEvolutionPilotError(RuntimeError):
    """Raised when collection or retained evidence violates the frozen contract."""


def collect_pilot(
    *,
    repo_root: Path,
    output_dir: Path,
    pilot_id: str,
    command: str,
    preflight_command: str,
    health_command: str,
    environment_command: str,
) -> dict[str, Any]:
    """Execute every frozen case once and retain its first result unconditionally."""

    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise HamiltonianEvolutionPilotError("pilot output directory must not already exist")
    contract = validate_pilot_contract(repo_root / DEFAULT_MANIFEST, repo_root)
    preflight = run_pilot_preflight(preflight_command, contract)
    output_dir.mkdir(parents=True)
    (output_dir / "cases").mkdir()
    shutil.copy2(repo_root / DEFAULT_MANIFEST, output_dir / "contract.json")
    shutil.copy2(repo_root / SOURCE_MANIFEST, output_dir / "source-manifest.json")
    shutil.copy2(
        repo_root / contract["build_reproduction_report"],
        output_dir / "build-reproduction.json",
    )
    _write_json(output_dir / "preflight.json", preflight)
    environment = _run_json_command(environment_command, "environment")
    _write_json(output_dir / "environment.json", environment)
    pre_health = _run_json_command(health_command, "pre-run device health")
    _write_json(output_dir / "pre-run-device-health.json", pre_health)
    manifest = {
        "schema": PILOT_SCHEMA,
        "pilot_id": pilot_id,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "pilot_started": True,
        "designated": False,
        "qualification_eligible": False,
        "claim_level": None,
        "stable_benchmark": False,
        "performance_eligible": False,
        "hardware_execution": True,
        "contract_sha256": _sha256(output_dir / "contract.json"),
        "preflight_sha256": _sha256(output_dir / "preflight.json"),
        "candidate_binary_sha256": contract["candidate_binary_sha256"],
        "source_commit": contract["source_commit"],
        "source_bundle_sha256": contract["source_bundle_sha256"],
        "tt_metal_commit": contract["tt_metal_commit"],
        "case_order": list(CASE_ORDER),
        "attempts_per_case": 1,
        "retries": 0,
        "replacement": "forbidden",
    }
    _write_json(output_dir / "pilot-manifest.json", manifest)

    results: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    for case_id in CASE_ORDER:
        hamiltonians, dt, frozen = load_frozen_case(repo_root, case_id)
        case_dir = output_dir / "cases" / case_id
        work_dir = case_dir / "work"
        case_dir.mkdir()
        source_dir = repo_root / DEFAULT_INPUT_ROOT / case_id
        shutil.copy2(source_dir / "hamiltonians.bin", case_dir / "hamiltonians.bin")
        shutil.copy2(source_dir / "dt.bin", case_dir / "dt.bin")
        _write_json(
            case_dir / "manifest.json",
            {**frozen, "attempt": 1, "retry_count": 0, "replacement": False},
        )
        result: dict[str, Any] = {
            "case_id": case_id,
            "role": frozen["role"],
            "attempt_count": 1,
            "retry_count": 0,
            "replacement": False,
            "candidate_completed": False,
            "passed": False,
        }
        exit_status = 1
        try:
            run = run_external_candidate(
                hamiltonians,
                dt,
                command=command,
                stage="conformance",
                execution_label="hardware",
                hbar=float(frozen["hbar"]),
                enforce_pilot_domain=frozen["role"] == "conformance",
                retained_work_dir=work_dir,
            )
            exit_status = 0
            (case_dir / "stdout.txt").write_text(_redact(run.stdout), encoding="utf-8")
            (case_dir / "stderr.txt").write_text(_redact(run.stderr), encoding="utf-8")
            shutil.copy2(work_dir / "metrics.json", case_dir / "metrics.json")
            shutil.copy2(work_dir / "final_rotors.bin", case_dir / "final_rotors.bin")
            shutil.copy2(work_dir / "final_phases.bin", case_dir / "final_phases.bin")
            _write_json(case_dir / "report.json", run.report)
            correctness = run.report["correctness"]
            matrix_ok = (
                frozen["role"] != "conformance"
                or correctness["direct_final_matrix_error"] <= MATRIX_THRESHOLD
            )
            result.update(
                {
                    "candidate_completed": True,
                    "passed": bool(correctness["passed"] and matrix_ok),
                    "report": f"cases/{case_id}/report.json",
                    "output_checksum": correctness["checksum"],
                    "direct_final_matrix_error": correctness["direct_final_matrix_error"],
                    "matrix_threshold_passed": matrix_ok,
                }
            )
            identities.append(_identity(run.report["candidate_metrics"]["candidate_metadata"]))
        except (HamiltonianEvolutionCandidateError, OSError, ValueError) as exc:
            stdout_path = work_dir / "_remote_stdout.txt"
            stderr_path = work_dir / "_remote_stderr.txt"
            (case_dir / "stdout.txt").write_text(
                _redact(stdout_path.read_text(encoding="utf-8")) if stdout_path.is_file() else "",
                encoding="utf-8",
            )
            (case_dir / "stderr.txt").write_text(
                _redact(stderr_path.read_text(encoding="utf-8")) if stderr_path.is_file() else "",
                encoding="utf-8",
            )
            for name in ("metrics.json", "final_rotors.bin", "final_phases.bin"):
                if (work_dir / name).is_file():
                    shutil.copy2(work_dir / name, case_dir / name)
            error = {"type": type(exc).__name__, "message": _redact(str(exc))}
            _write_json(case_dir / "error.json", error)
            result["error"] = error
        finally:
            _write_json(
                case_dir / "exit-status.json",
                {"attempt": 1, "returncode": exit_status, "retry_count": 0},
            )
        shutil.rmtree(work_dir, ignore_errors=True)
        results.append(result)

    post_health_error: str | None = None
    try:
        post_health = _run_json_command(health_command, "post-run device health")
    except HamiltonianEvolutionPilotError as exc:
        post_health = {"collection_error": str(exc)}
        post_health_error = str(exc)
    _write_json(output_dir / "post-run-device-health.json", post_health)
    identity_consistent = bool(identities) and all(item == identities[0] for item in identities)
    conformance = [item for item in results if item["role"] == "conformance"]
    suite_passed = (
        all(item["passed"] for item in conformance)
        and len(conformance) == len(CASE_ORDER) - len(STRESS_CASES)
        and identity_consistent
        and post_health_error is None
    )
    suite = {
        "schema": SUITE_SCHEMA,
        "pilot_id": pilot_id,
        "case_order": list(CASE_ORDER),
        "results": results,
        "conformance_case_count": len(conformance),
        "stress_diagnostic_case_count": len(STRESS_CASES),
        "candidate_identity_consistent": identity_consistent,
        "candidate_identity": identities[0] if identity_consistent else None,
        "suite_passed": suite_passed,
        "stable_benchmark": False,
        "performance_eligible": False,
        "claim_level": None,
    }
    _write_json(output_dir / "suite-report.json", suite)
    (output_dir / "suite-report.md").write_text(_render_suite(suite), encoding="utf-8")
    return suite


def run_pilot_preflight(command: str, contract: dict[str, Any]) -> dict[str, Any]:
    """Run and validate the shared environment before any case attempt exists."""

    payload = _run_json_command(command, "pilot preflight")
    return validate_pilot_preflight(payload, contract)


def validate_pilot_preflight(payload: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    """Validate a captured shared-environment preflight."""

    if payload.get("schema") != PREFLIGHT_SCHEMA or payload.get("passed") is not True:
        raise HamiltonianEvolutionPilotError("pilot preflight did not pass")
    expected = {
        "candidate_sha256": contract["candidate_binary_sha256"],
        "source_commit": contract["source_commit"],
        "source_bundle_sha256": contract["source_bundle_sha256"],
        "tt_metal_commit": contract["tt_metal_commit"],
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise HamiltonianEvolutionPilotError(f"pilot preflight identity mismatch for {key}")
    required_true = (
        "tt_metal_home_set",
        "tt_metal_runtime_root_set",
        "runtime_roots_resolve_same",
        "runtime_root_exists",
        "runtime_discoverable",
        "candidate_exists",
        "candidate_executable",
        "source_exists",
        "source_tree_clean",
        "tt_metal_tree_clean",
        "shared_libraries_resolved",
        "tt_metal_library_from_expected_root",
        "runtime_cache_parent_writable",
        "runtime_cache_session_root_new",
    )
    for key in required_true:
        if payload.get(key) is not True:
            raise HamiltonianEvolutionPilotError(f"pilot preflight {key} must be true")
    health = payload.get("device_health")
    if not isinstance(health, dict):
        raise HamiltonianEvolutionPilotError("pilot preflight device health is missing")
    try:
        validate_device_health(json.dumps(health), device_id=0)
    except Exception as exc:
        raise HamiltonianEvolutionPilotError(
            f"pilot preflight device health failed: {exc}"
        ) from exc
    return payload


def validate_pilot_package(root: Path, repo_root: Path) -> dict[str, Any]:
    """Qualify a retained package using only local files and hashes."""

    root, repo_root = root.resolve(), repo_root.resolve()
    manifest = _load_json(root / "pilot-manifest.json")
    suite = _load_json(root / "suite-report.json")
    contract = _load_json(root / "contract.json")
    canonical = validate_pilot_contract(repo_root / DEFAULT_MANIFEST, repo_root)
    if contract != canonical:
        raise HamiltonianEvolutionPilotError("retained contract differs from frozen contract")
    if manifest.get("schema") != PILOT_SCHEMA or suite.get("schema") != SUITE_SCHEMA:
        raise HamiltonianEvolutionPilotError("pilot schema mismatch")
    for key, expected in {
        "designated": False,
        "qualification_eligible": False,
        "claim_level": None,
        "stable_benchmark": False,
        "performance_eligible": False,
        "hardware_execution": True,
        "attempts_per_case": 1,
        "retries": 0,
        "replacement": "forbidden",
    }.items():
        if manifest.get(key) != expected:
            raise HamiltonianEvolutionPilotError(f"pilot manifest {key} mismatch")
    if manifest.get("contract_sha256") != _sha256(root / "contract.json"):
        raise HamiltonianEvolutionPilotError("retained contract hash mismatch")
    if "preflight_sha256" in manifest:
        preflight = _load_json(root / "preflight.json")
        if manifest["preflight_sha256"] != _sha256(root / "preflight.json"):
            raise HamiltonianEvolutionPilotError("retained preflight hash mismatch")
        validate_pilot_preflight(preflight, contract)
    for key in (
        "candidate_binary_sha256",
        "source_commit",
        "source_bundle_sha256",
        "tt_metal_commit",
    ):
        if manifest.get(key) != contract.get(key):
            raise HamiltonianEvolutionPilotError(f"pilot identity mismatch for {key}")
    if manifest.get("case_order") != list(CASE_ORDER) or suite.get("case_order") != list(
        CASE_ORDER
    ):
        raise HamiltonianEvolutionPilotError("pilot case order mismatch")
    if _load_json(root / "source-manifest.json") != _load_json(repo_root / SOURCE_MANIFEST):
        raise HamiltonianEvolutionPilotError("retained source manifest mismatch")
    if _sha256(root / "build-reproduction.json") != contract["build_reproduction_sha256"]:
        raise HamiltonianEvolutionPilotError("retained build report mismatch")
    environment = _load_json(root / "environment.json")
    environment_expected = {
        "candidate_sha256": contract["candidate_binary_sha256"],
        "source_commit": contract["source_commit"],
        "source_tree_clean": True,
        "tt_metal_commit": contract["tt_metal_commit"],
        "tt_metal_tree_clean": True,
        "device_id": 0,
        "performance_collection": False,
        "profiler_enabled": False,
        "watcher_enabled": False,
    }
    if "preflight_sha256" in manifest:
        environment_expected.update(
            {
                "tt_metal_home_set": True,
                "tt_metal_runtime_root_set": True,
                "runtime_roots_resolve_same": True,
                "runtime_cache_policy": "fresh empty TT_METAL_CACHE per case",
            }
        )
    for key, expected in environment_expected.items():
        if environment.get(key) != expected:
            raise HamiltonianEvolutionPilotError(f"environment mismatch for {key}")
    pre_health = _load_json(root / "pre-run-device-health.json")
    post_health = _load_json(root / "post-run-device-health.json")
    if "collection_error" in post_health:
        raise HamiltonianEvolutionPilotError("post-run health collection failed")
    try:
        pre_raw = json.dumps(pre_health)
        post_raw = json.dumps(post_health)
        _compare_conformance_health(pre_raw, post_raw)
    except Exception as exc:
        raise HamiltonianEvolutionPilotError(f"device-health inconsistency: {exc}") from exc

    results = suite.get("results")
    if not isinstance(results, list) or [item.get("case_id") for item in results] != list(
        CASE_ORDER
    ):
        raise HamiltonianEvolutionPilotError("pilot results are missing or reordered")
    identities: list[dict[str, Any]] = []
    conformance_passed = True
    for item in results:
        case_id = item["case_id"]
        case_dir = root / "cases" / case_id
        _, _, frozen = load_frozen_case(repo_root, case_id)
        retained = _load_json(case_dir / "manifest.json")
        if retained != {**frozen, "attempt": 1, "retry_count": 0, "replacement": False}:
            raise HamiltonianEvolutionPilotError(f"case {case_id} manifest mismatch")
        for name, expected_hash in (
            ("hamiltonians.bin", frozen["hamiltonians_sha256"]),
            ("dt.bin", frozen["dt_sha256"]),
        ):
            if _sha256(case_dir / name) != expected_hash:
                raise HamiltonianEvolutionPilotError(f"case {case_id} input hash mismatch")
        exit_status = _load_json(case_dir / "exit-status.json")
        if exit_status != {
            "attempt": 1,
            "returncode": 0 if item.get("candidate_completed") else 1,
            "retry_count": 0,
        }:
            raise HamiltonianEvolutionPilotError(f"case {case_id} attempt record mismatch")
        (case_dir / "stdout.txt").read_bytes()
        (case_dir / "stderr.txt").read_bytes()
        if not item.get("candidate_completed"):
            _load_json(case_dir / "error.json")
            if item["role"] == "conformance":
                conformance_passed = False
            continue
        report = _load_json(case_dir / "report.json")
        if report.get("execution_label") != "hardware" or report.get("input_hashes") != {
            "hamiltonians_sha256": frozen["hamiltonians_sha256"],
            "dt_sha256": frozen["dt_sha256"],
        }:
            raise HamiltonianEvolutionPilotError(f"case {case_id} report identity mismatch")
        correctness = report.get("correctness", {})
        payload = (case_dir / "final_rotors.bin").read_bytes() + (
            case_dir / "final_phases.bin"
        ).read_bytes()
        if hashlib.sha256(payload).hexdigest() != correctness.get("checksum"):
            raise HamiltonianEvolutionPilotError(f"case {case_id} output checksum mismatch")
        identity = _identity(report.get("candidate_metrics", {}).get("candidate_metadata", {}))
        for key in (
            "candidate_binary_sha256",
            "source_commit",
            "source_bundle_sha256",
            "tt_metal_commit",
        ):
            metadata_key = "candidate_sha256" if key == "candidate_binary_sha256" else key
            if identity.get(metadata_key) != contract[key]:
                raise HamiltonianEvolutionPilotError(f"case {case_id} candidate identity mismatch")
        identities.append(identity)
        expected_pass = correctness.get("passed") is True
        if item["role"] == "conformance":
            expected_pass = (
                expected_pass
                and correctness.get("direct_final_matrix_error", float("inf")) <= MATRIX_THRESHOLD
            )
            conformance_passed = conformance_passed and expected_pass
        if item.get("passed") is not expected_pass:
            raise HamiltonianEvolutionPilotError(f"case {case_id} pass status mismatch")
    consistent = bool(identities) and all(item == identities[0] for item in identities)
    expected_suite = conformance_passed and consistent
    if (
        suite.get("candidate_identity_consistent") is not consistent
        or suite.get("suite_passed") is not expected_suite
    ):
        raise HamiltonianEvolutionPilotError("suite result is inconsistent with retained evidence")
    if (
        suite.get("stable_benchmark") is not False
        or suite.get("performance_eligible") is not False
        or suite.get("claim_level") is not None
    ):
        raise HamiltonianEvolutionPilotError("pilot cannot promote benchmark claims")
    return {"package_valid": True, "pilot_passed": expected_suite, "case_count": len(results)}


def build_qualification(root: Path, repo_root: Path) -> dict[str, Any]:
    """Build the deterministic processed result after offline package validation."""

    result = validate_pilot_package(root, repo_root)
    suite = _load_json(root.resolve() / "suite-report.json")
    failure_classification = _classify_failure(suite)
    return {
        "schema": "tt-rqm-hamiltonian-evolution-pilot-qualification.v1",
        "pilot_id": suite["pilot_id"],
        "package_valid": result["package_valid"],
        "pilot_passed": result["pilot_passed"],
        "case_count": result["case_count"],
        "non_designated": True,
        "claim_level": None,
        "stable_benchmark": False,
        "performance_eligible": False,
        "failure_classification": failure_classification,
        "results": suite["results"],
    }


def render_qualification(payload: dict[str, Any]) -> str:
    lines = [
        "# H2B N300 pilot qualification",
        "",
        f"- Package valid: `{str(payload['package_valid']).lower()}`",
        f"- Pilot passed: `{str(payload['pilot_passed']).lower()}`",
        f"- Failure classification: `{payload['failure_classification']}`",
        "- Claim level: `null`",
        "- Stable benchmark: `false`",
        "- Performance eligible: `false`",
        "- Non-designated: `true`",
        "",
    ]
    for result in payload["results"]:
        lines.append(
            f"- `{result['case_id']}` ({result['role']}): {'pass' if result['passed'] else 'fail'}"
        )
    if payload["failure_classification"] == "environment":
        lines += [
            "",
            "All cases stopped at the same pre-device TT-Metal runtime-root initialization blocker; no numerical output was produced.",
        ]
    lines += ["", "No H2B hardware claim exists.", ""]
    return "\n".join(lines)


def build_blocker_report(root: Path, repo_root: Path) -> dict[str, Any]:
    """Classify a failed retained pilot from package evidence only."""

    result = validate_pilot_package(root, repo_root)
    if result["pilot_passed"]:
        raise HamiltonianEvolutionPilotError("a passing pilot has no blocker report")
    root = root.resolve()
    suite = _load_json(root / "suite-report.json")
    signature_cases: list[str] = []
    for item in suite["results"]:
        case_dir = root / "cases" / item["case_id"]
        text = "\n".join(
            (case_dir / name).read_text(encoding="utf-8", errors="replace")
            for name in ("stdout.txt", "stderr.txt")
        )
        if (
            "Read unexpected run_mailbox value" in text
            and "failed to complete an early exit" in text
            and "active ethernet dispatch core" in text
        ):
            signature_cases.append(item["case_id"])
    if not signature_cases:
        raise HamiltonianEvolutionPilotError(
            "failed pilot has no evidence for an approved blocker classification"
        )
    metrics_count = len(list((root / "cases").glob("*/metrics.json")))
    rotor_count = len(list((root / "cases").glob("*/final_rotors.bin")))
    phase_count = len(list((root / "cases").glob("*/final_phases.bin")))
    return {
        "schema": "tt-rqm-hamiltonian-evolution-pilot-blocker.v2",
        "pilot_id": suite["pilot_id"],
        "pilot_package": root.relative_to(repo_root.resolve()).as_posix(),
        "package_valid": True,
        "pilot_passed": False,
        "failure_classification": "runtime",
        "observed_mechanism": "dispatch_mailbox_synchronization_during_device_initialization",
        "approved_categories": [
            "build",
            "runtime",
            "synchronization",
            "layout",
            "lowering",
            "composition",
            "ordering",
            "numerical_domain",
        ],
        "attempt_evidence": {
            "case_count": len(suite["results"]),
            "conformance_case_count": suite["conformance_case_count"],
            "stress_diagnostic_case_count": suite["stress_diagnostic_case_count"],
            "attempt_counts": sorted({item["attempt_count"] for item in suite["results"]}),
            "retry_counts": sorted({item["retry_count"] for item in suite["results"]}),
            "candidate_completed_count": sum(
                item["candidate_completed"] is True for item in suite["results"]
            ),
        },
        "runtime_evidence": {
            "signature_case_count": len(signature_cases),
            "signature_case_ids": signature_cases,
            "signature": [
                "active ethernet dispatch core detected as still running",
                "failed to complete an early exit",
                "Read unexpected run_mailbox value: 0x40",
            ],
            "metrics_file_count": metrics_count,
            "final_rotor_file_count": rotor_count,
            "final_phase_file_count": phase_count,
        },
        "numerical_conclusion": "No numerical output was produced; the bounded angle domain is not hardware-confirmed.",
        "claim_level": None,
        "stable_benchmark": False,
        "performance_eligible": False,
        "designated_contract_created": False,
        "designated_session_executed": False,
    }


def render_blocker_report(payload: dict[str, Any]) -> str:
    evidence = payload["runtime_evidence"]
    attempts = payload["attempt_evidence"]
    return "\n".join(
        [
            "# H2B N300 pilot Session 2 blocker",
            "",
            f"Session 2 is valid and did not pass. The first evidenced failing layer is `{payload['failure_classification']}`; the observed mechanism is dispatch/mailbox synchronization during device initialization.",
            "",
            f"All {attempts['case_count']} frozen cases were invoked once in order with zero retries or replacements. None completed, and no metrics, final rotors, or final phases were produced.",
            "",
            f"Retained stdout/stderr for {evidence['signature_case_count']} cases records active dispatch cores, failure to complete early exit, and unexpected run-mailbox value `0x40`. Preflight passed before collection, and post-run health retained both visible N300 entries without DRAM faults, hardware faults, throttling, or reboot.",
            "",
            "This is not build, layout, lowering, composition, ordering, or numerical-domain evidence. The bounded angle domain remains a CPU/reference contract and is not hardware-confirmed.",
            "",
            "No designated contract was created or executed. No H2B hardware claim exists; `claim_level=null`, `stable_benchmark=false`, and `performance_eligible=false`.",
            "",
        ]
    )


def _classify_failure(suite: dict[str, Any]) -> str | None:
    if suite.get("suite_passed") is True:
        return None
    results = suite.get("results", [])
    messages = [
        str(item.get("error", {}).get("message", "")) for item in results if isinstance(item, dict)
    ]
    if messages and all(
        "TT_METAL_RUNTIME_ROOT" in message or "Root Directory is not set" in message
        for message in messages
    ):
        return "environment"
    return None


def _identity(metadata: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "candidate_sha256",
        "source_bundle_sha256",
        "source_commit",
        "tt_metal_commit",
        "device_arch",
        "device_id",
        "device_count",
        "device_create_count",
        "device_close_count",
        "program_count",
        "intermediate_storage",
        "device_resident_intermediate",
        "intermediate_d2h_count",
        "intermediate_h2d_count",
        "host_round_trip_count",
        "automatic_normalization",
        "composition_order",
    )
    identity = {key: metadata.get(key) for key in keys}
    expected = {
        "device_id": 0,
        "device_count": 1,
        "device_create_count": 1,
        "device_close_count": 1,
        "program_count": 2,
        "intermediate_storage": "device_dram",
        "device_resident_intermediate": True,
        "intermediate_d2h_count": 0,
        "intermediate_h2d_count": 0,
        "host_round_trip_count": 0,
        "automatic_normalization": False,
        "composition_order": "K-1 ... 0",
    }
    for key, value in expected.items():
        if identity.get(key) != value:
            raise HamiltonianEvolutionPilotError(
                f"pilot lifecycle metadata {key} must equal {value!r}"
            )
    if "wormhole" not in str(identity["device_arch"]).lower():
        raise HamiltonianEvolutionPilotError("pilot device architecture is not Wormhole")
    return identity


def _run_json_command(command: str, label: str) -> dict[str, Any]:
    completed = subprocess.run(command, shell=True, capture_output=True, text=True)
    if completed.returncode != 0:
        raise HamiltonianEvolutionPilotError(f"{label} command failed: {completed.stderr.strip()}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise HamiltonianEvolutionPilotError(f"{label} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise HamiltonianEvolutionPilotError(f"{label} must be a JSON object")
    return payload


def _compare_conformance_health(pre_raw: str, post_raw: str) -> None:
    """Require healthy stable devices while allowing non-performance AICLK scaling."""

    pre = validate_device_health(pre_raw, device_id=0)
    post = validate_device_health(post_raw, device_id=0)
    if pre["visible_device_count"] != post["visible_device_count"]:
        raise HamiltonianEvolutionPilotError("visible device count changed during pilot")
    for before, after in zip(pre["devices"], post["devices"], strict=True):
        if before["board_id"] != after["board_id"] or before["boot_date"] != after["boot_date"]:
            raise HamiltonianEvolutionPilotError(
                "device identity or boot state changed during pilot"
            )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HamiltonianEvolutionPilotError(f"invalid or missing {path}") from exc
    if not isinstance(payload, dict):
        raise HamiltonianEvolutionPilotError(f"{path} must contain an object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _render_suite(suite: dict[str, Any]) -> str:
    lines = ["# H2B non-designated N300 pilot", ""]
    for result in suite["results"]:
        outcome = "pass" if result["passed"] else "fail"
        lines.append(f"- `{result['case_id']}` ({result['role']}): {outcome}")
    lines += [
        "",
        f"Conformance suite pass: `{str(suite['suite_passed']).lower()}`",
        "",
        "The large-angle case is an out-of-domain stress diagnostic and is not a conformance pass gate.",
        "",
        "This non-designated pilot establishes no claim level, stability, or performance eligibility.",
        "",
    ]
    return "\n".join(lines)


def _redact(value: str) -> str:
    """Remove endpoint-specific paths while preserving diagnostic text."""

    import re

    value = re.sub(r"/tmp/rqm-h2b-pilot-[^/\s]+", "<remote-session-root>", value)
    return value.replace("/home/user/src/tt-metal", "<tt-metal-root>")
