from __future__ import annotations

import json
from pathlib import Path
import shlex
import shutil
import sys

import pytest

from tt_rqm_kernels.hamiltonian_evolution_pilot import (
    HamiltonianEvolutionPilotError,
    _identity,
    build_blocker_report,
    collect_pilot,
    validate_pilot_preflight,
    validate_pilot_package,
)
from tt_rqm_kernels.hamiltonian_evolution_pilot_contract import (
    DEFAULT_MANIFEST,
    validate_pilot_contract,
)

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "benchmarks/pilots/hamiltonian-evolution-h2b/h2b-n300-pilot-20260716-session-1"
SESSION_2 = ROOT / "benchmarks/pilots/hamiltonian-evolution-h2b/h2b-n300-pilot-20260716-session-2"


def test_retained_first_pilot_is_valid_and_failed() -> None:
    assert validate_pilot_package(PILOT, ROOT) == {
        "package_valid": True,
        "pilot_passed": False,
        "case_count": 20,
    }
    suite = json.loads((PILOT / "suite-report.json").read_text())
    assert len(suite["results"]) == 20
    assert all(item["attempt_count"] == 1 for item in suite["results"])
    assert all(item["retry_count"] == 0 for item in suite["results"])
    assert all(item["candidate_completed"] is False for item in suite["results"])


def test_retained_session_2_is_valid_and_classified_separately() -> None:
    assert validate_pilot_package(SESSION_2, ROOT) == {
        "package_valid": True,
        "pilot_passed": False,
        "case_count": 20,
    }
    blocker = build_blocker_report(SESSION_2, ROOT)
    assert blocker["failure_classification"] == "runtime"
    assert blocker["runtime_evidence"]["signature_case_count"] == 19
    assert blocker["runtime_evidence"]["metrics_file_count"] == 0
    assert blocker["designated_contract_created"] is False


def test_offline_qualifier_rejects_retry_or_reordered_case(tmp_path: Path) -> None:
    copied = tmp_path / "pilot"
    shutil.copytree(PILOT, copied)
    manifest_path = copied / "pilot-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["retries"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(HamiltonianEvolutionPilotError, match="retries"):
        validate_pilot_package(copied, ROOT)

    shutil.rmtree(copied)
    shutil.copytree(PILOT, copied)
    suite_path = copied / "suite-report.json"
    suite = json.loads(suite_path.read_text())
    suite["results"] = list(reversed(suite["results"]))
    suite_path.write_text(json.dumps(suite), encoding="utf-8")
    with pytest.raises(HamiltonianEvolutionPilotError, match="missing or reordered"):
        validate_pilot_package(copied, ROOT)


def test_invalid_preflight_creates_no_session_or_case_records(tmp_path: Path) -> None:
    output = tmp_path / "session-2"
    invalid = shlex.join(
        [sys.executable, "-c", "import json; print(json.dumps({'passed': False}))"]
    )
    with pytest.raises(HamiltonianEvolutionPilotError, match="did not pass"):
        collect_pilot(
            repo_root=ROOT,
            output_dir=output,
            pilot_id="test-session-2",
            command="unused",
            preflight_command=invalid,
            health_command="unused",
            environment_command="unused",
        )
    assert not output.exists()


def test_valid_preflight_binds_identities_and_n300_health() -> None:
    contract = validate_pilot_contract(ROOT / DEFAULT_MANIFEST, ROOT)
    health = json.loads((PILOT / "pre-run-device-health.json").read_text())
    payload = {
        "schema": "tt-rqm-hamiltonian-evolution-pilot-preflight.v1",
        "passed": True,
        "candidate_sha256": contract["candidate_binary_sha256"],
        "source_commit": contract["source_commit"],
        "source_bundle_sha256": contract["source_bundle_sha256"],
        "tt_metal_commit": contract["tt_metal_commit"],
        "device_health": health,
    }
    for key in (
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
    ):
        payload[key] = True
    assert validate_pilot_preflight(payload, contract) is payload


def test_lifecycle_metadata_requires_exact_two_program_device_resident_execution() -> None:
    metadata = {
        "candidate_sha256": "a" * 64,
        "source_bundle_sha256": "b" * 64,
        "source_commit": "c" * 40,
        "tt_metal_commit": "d" * 40,
        "device_arch": "wormhole_b0",
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
    assert _identity(metadata)["program_count"] == 2
    for key, invalid in (
        ("device_create_count", 2),
        ("device_close_count", 0),
        ("program_count", 3),
        ("intermediate_d2h_count", 1),
        ("intermediate_h2d_count", 1),
        ("host_round_trip_count", 1),
        ("automatic_normalization", True),
        ("composition_order", "0 ... K-1"),
    ):
        tampered = {**metadata, key: invalid}
        with pytest.raises(HamiltonianEvolutionPilotError, match=key):
            _identity(tampered)
