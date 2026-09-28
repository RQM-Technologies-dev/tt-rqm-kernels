"""Depth-chained proof-gated SU(4) conformance support."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from tt_rqm_kernels.su4q_conformance import (
    ATOL,
    BLOCK_CONVENTION,
    BLOCK_LANES,
    RQMC_COMMIT,
    RQME_COMMIT,
    RTOL,
    STATE_LANES,
    TT_METAL_COMMIT,
    TT_RQM_COMMIT,
    _compiler_cases,
    _require_base,
    _require_pin,
    _sentinel_cases,
    block_to_unitary,
    cartan_core,
    git_commit,
    quaternion_to_su2,
    routing_contract_checks,
    sha256_file,
)


PROTOCOL = "tt-rqm-su4q-chain-conformance.v1"
METRICS_SCHEMA = "tt-rqm-su4q-chain-conformance-metrics.v1"
VALIDATION_SCHEMA = "tt-rqm-su4q-chain-conformance-validation.v1"
DEPTHS = (1, 8, 32, 128)
ITEMS = 128


def _direct_chain_cases() -> list[dict[str, Any]]:
    identity_q = np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64)
    local_q0 = np.asarray((0.81, 0.21, -0.37, 0.39), dtype=np.float64)
    local_q0 /= np.linalg.norm(local_q0)
    local_q1 = np.asarray((0.73, -0.29, 0.42, 0.46), dtype=np.float64)
    local_q1 /= np.linalg.norm(local_q1)
    local_block = np.asarray(
        (*local_q0, *local_q1, 0.0, 0.0, 0.0, *identity_q, *identity_q, 0.0),
        dtype=np.float64,
    )
    angle = 0.31
    positive_block = np.asarray(
        (*identity_q, *identity_q, angle, 0.0, 0.0, *identity_q, *identity_q, 0.0),
        dtype=np.float64,
    )
    negative_block = positive_block.copy()
    negative_block[8] = -angle
    return [
        {
            "id": "chain_local_only",
            "kind": "chain_sentinel",
            "block": local_block,
            "unitary": np.kron(
                quaternion_to_su2(local_q1), quaternion_to_su2(local_q0)
            ),
            "oracle_source": "explicit_local_matrix",
        },
        {
            "id": "chain_inverse_positive",
            "kind": "chain_sentinel",
            "block": positive_block,
            "unitary": cartan_core(angle, 0.0, 0.0),
            "oracle_source": "explicit_positive_cartan_matrix",
        },
        {
            "id": "chain_inverse_negative",
            "kind": "chain_sentinel",
            "block": negative_block,
            "unitary": cartan_core(-angle, 0.0, 0.0),
            "oracle_source": "explicit_negative_cartan_matrix",
        },
    ]


def _case_pool() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    compiler, routing = _compiler_cases()
    for case in compiler:
        case["oracle_source"] = "original_source_circuit_complex128"
    sentinels = _sentinel_cases()
    for case in sentinels:
        case["oracle_source"] = "explicit_convention_matrix"
    return compiler + sentinels + _direct_chain_cases(), routing


def _case_for(
    trajectory: int, layer: int, cases: list[dict[str, Any]]
) -> tuple[int, str]:
    compiler_count = 6
    by_id = {case["id"]: index for index, case in enumerate(cases)}
    scenario = trajectory % 16
    if scenario < compiler_count:
        return scenario, f"repeated_{cases[scenario]['id']}"
    if scenario == 6:
        return (trajectory + layer) % compiler_count, "mixed_compiler_forward"
    if scenario == 7:
        return (trajectory - layer) % compiler_count, "mixed_compiler_reverse"
    if scenario == 8:
        ids = ("sentinel_q1_tensor_q0_order", "sentinel_positive_cartan_sign")
        return by_id[ids[layer % 2]], "noncommutative_ab"
    if scenario == 9:
        ids = ("sentinel_positive_cartan_sign", "sentinel_q1_tensor_q0_order")
        return by_id[ids[layer % 2]], "noncommutative_ba"
    if scenario == 10:
        ids = ("chain_inverse_positive", "chain_inverse_negative")
        return by_id[ids[layer % 2]], "inverse_cancellation"
    if scenario == 11:
        return by_id["chain_local_only"], "local_only"
    if scenario == 12:
        return by_id["sentinel_global_phase"], "global_phase_accumulation"
    if scenario == 13:
        return by_id["sentinel_identity"], "identity_chain"
    if scenario == 14:
        return by_id["sentinel_iswap_class"], "iswap_chain"
    return (trajectory + 5 * layer) % len(cases), "full_pool_mixed"


def prepare_session(
    output_dir: Path,
    *,
    depth: int,
    compiler_root: Path,
    entanglement_root: Path,
    tt_rqm_root: Path,
    tt_metal_root: Path,
) -> Path:
    if depth not in DEPTHS:
        raise ValueError(f"depth must be one of {DEPTHS}")
    for root, expected, label in (
        (compiler_root, RQMC_COMMIT, "rqm-compiler"),
        (entanglement_root, RQME_COMMIT, "rqm-entanglement"),
        (tt_metal_root, TT_METAL_COMMIT, "tt-metal"),
    ):
        _require_pin(root.resolve(), expected, label)
    _require_base(tt_rqm_root.resolve(), TT_RQM_COMMIT, "tt-rqm-kernels")
    output_dir.mkdir(parents=True, exist_ok=False)
    cases, routing = _case_pool()
    checks = routing_contract_checks()
    rng = np.random.default_rng(20260727 + depth)
    initial = np.empty((ITEMS, 4), dtype=np.complex128)
    basis = np.eye(4, dtype=np.complex128)
    for trajectory in range(ITEMS):
        if trajectory % 8 < 4:
            initial[trajectory] = basis[trajectory % 4]
        else:
            value = rng.normal(size=4) + 1j * rng.normal(size=4)
            initial[trajectory] = value / np.linalg.norm(value)
    blocks = np.empty((depth, ITEMS, 20), dtype=np.float32)
    case_indices = np.empty((depth, ITEMS), dtype=np.uint32)
    scenario_ids: list[str] = []
    final = initial.copy()
    for layer in range(depth):
        for trajectory in range(ITEMS):
            case_index, scenario = _case_for(trajectory, layer, cases)
            if layer == 0 and trajectory < len(cases):
                case_index = trajectory
            if layer == 0:
                scenario_ids.append(scenario)
            case = cases[case_index]
            blocks[layer, trajectory] = case["block"].astype(np.float32)
            case_indices[layer, trajectory] = case_index
            final[trajectory] = case["unitary"] @ final[trajectory]
    states = np.empty((ITEMS, 8), dtype=np.float32)
    states[:, 0::2] = initial.real.astype(np.float32)
    states[:, 1::2] = initial.imag.astype(np.float32)
    references = np.empty((ITEMS, 8), dtype=np.float64)
    references[:, 0::2] = final.real
    references[:, 1::2] = final.imag
    scenario_indices = np.asarray(
        [sorted(set(scenario_ids)).index(value) for value in scenario_ids],
        dtype=np.uint32,
    )
    paths = {
        "blocks": output_dir / "blocks.bin",
        "states": output_dir / "states.bin",
        "case_indices": output_dir / "case_indices.bin",
        "scenario_indices": output_dir / "scenario_indices.bin",
        "reference_states": output_dir / "reference_states.bin",
    }
    for name, values in (
        ("blocks", blocks),
        ("states", states),
        ("case_indices", case_indices),
        ("scenario_indices", scenario_indices),
        ("reference_states", references),
    ):
        values.tofile(paths[name])
    manifest = {
        "schema": PROTOCOL,
        "experiment": "su4q-chain-n300-conformance",
        "stage": "conformance",
        "dtype": "float32",
        "performance_eligible": False,
        "items": ITEMS,
        "depth": depth,
        "block_shape": [depth, ITEMS, 20],
        "state_shape": [ITEMS, 8],
        "block_lanes": list(BLOCK_LANES),
        "state_lanes": list(STATE_LANES),
        "block_convention_version": BLOCK_CONVENTION,
        "cartan_convention": "exp[i(aXX+bYY+cZZ)]",
        "local_factor_order": "SU(2)(q1) tensor SU(2)(q0)",
        "basis_order": ["00", "01", "10", "11"],
        "atol": ATOL,
        "rtol": RTOL,
        "inputs": {
            name: {"file": path.name, "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "outputs": {
            "states": "output_states.bin",
            "metrics": "metrics.json",
            "validation": "validation.json",
        },
        "cases": [
            {
                "index": index,
                "id": case["id"],
                "kind": case["kind"],
                "oracle_source": case["oracle_source"],
            }
            for index, case in enumerate(cases)
        ],
        "scenarios": sorted(set(scenario_ids)),
        "compiler_routing": routing,
        "routing_contract_checks": checks,
        "provenance": {
            "rqm_compiler_commit": RQMC_COMMIT,
            "rqm_entanglement_commit": RQME_COMMIT,
            "tt_rqm_kernels_commit": TT_RQM_COMMIT,
            "tt_rqm_kernels_source_commit": git_commit(tt_rqm_root.resolve()),
            "tt_metal_commit": TT_METAL_COMMIT,
        },
    }
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def host_candidate(session_dir: Path) -> Path:
    manifest = json.loads((session_dir / "manifest.json").read_text())
    depth = int(manifest["depth"])
    blocks = np.fromfile(session_dir / "blocks.bin", dtype=np.float32).reshape(
        depth, ITEMS, 20
    )
    states = np.fromfile(session_dir / "states.bin", dtype=np.float32).reshape(ITEMS, 8)
    current = states[:, 0::2].astype(np.float64) + 1j * states[:, 1::2].astype(
        np.float64
    )
    for layer in range(depth):
        for trajectory in range(ITEMS):
            current[trajectory] = block_to_unitary(
                blocks[layer, trajectory].astype(np.float64)
            ) @ current[trajectory]
    output = np.empty((ITEMS, 8), dtype=np.float32)
    output[:, 0::2] = current.real.astype(np.float32)
    output[:, 1::2] = current.imag.astype(np.float32)
    path = session_dir / "host_output_states.bin"
    output.tofile(path)
    return path


def validate_session(session_dir: Path, *, output_name: str | None = None) -> dict[str, Any]:
    manifest = json.loads((session_dir / "manifest.json").read_text())
    if manifest.get("schema") != PROTOCOL:
        raise ValueError("unsupported SU4Q chain manifest")
    depth = int(manifest["depth"])
    if depth not in DEPTHS or manifest.get("items") != ITEMS:
        raise ValueError("invalid chain depth or item count")
    for record in manifest["inputs"].values():
        path = session_dir / record["file"]
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"input hash mismatch: {path.name}")
    output_path = session_dir / (
        output_name if output_name is not None else manifest["outputs"]["states"]
    )
    output = np.fromfile(output_path, dtype=np.float32)
    if output.size != ITEMS * 8:
        raise ValueError("candidate output has an unexpected element count")
    output = output.reshape(ITEMS, 8).astype(np.float64)
    reference = np.fromfile(session_dir / "reference_states.bin", dtype=np.float64).reshape(
        ITEMS, 8
    )
    error = np.abs(output - reference)
    threshold = ATOL + RTOL * np.abs(reference)
    nonfinite = int(np.count_nonzero(~np.isfinite(output)))
    failures = int(np.count_nonzero((error > threshold) | ~np.isfinite(output)))
    output_complex = output[:, 0::2] + 1j * output[:, 1::2]
    reference_complex = reference[:, 0::2] + 1j * reference[:, 1::2]
    overlaps = np.sum(np.conj(output_complex) * reference_complex, axis=1)
    phases = np.ones(ITEMS, dtype=np.complex128)
    nonzero = np.abs(overlaps) > 0
    phases[nonzero] = overlaps[nonzero] / np.abs(overlaps[nonzero])
    aligned = output_complex * phases[:, None]
    case_indices = np.fromfile(session_dir / "case_indices.bin", dtype=np.uint32)
    coverage = sorted(set(int(value) for value in case_indices))
    expected = list(range(len(manifest["cases"])))
    metrics_path = session_dir / manifest["outputs"]["metrics"]
    host_only = output_name is not None
    metadata: dict[str, Any] = {}
    provenance_match = True
    transfer_contract = True
    if not host_only:
        metrics = json.loads(metrics_path.read_text())
        metadata = metrics.get("candidate_metadata", {})
        provenance_match = (
            metrics.get("schema") == METRICS_SCHEMA
            and metrics.get("protocol") == PROTOCOL
            and metrics.get("performance_eligible") is False
            and metrics.get("depth") == depth
            and metadata.get("tt_metal_commit") == TT_METAL_COMMIT
            and metadata.get("device_id") == 0
            and metadata.get("device_count") == 1
            and metadata.get("device_create_count") == 1
            and metadata.get("device_close_count") == 1
            and metadata.get("program_count") == 8 * depth
            and metadata.get("source_tree_clean") is True
            and bool(metadata.get("candidate_sha256"))
            and bool(metadata.get("source_bundle_sha256"))
        )
        transfer_contract = (
            metadata.get("initial_block_upload_count") == 1
            and metadata.get("initial_state_upload_count") == 1
            and metadata.get("intermediate_h2d_count") == 0
            and metadata.get("intermediate_d2h_count") == 0
            and metadata.get("final_state_download_count") == 1
        )
    result = {
        "schema": VALIDATION_SCHEMA,
        "protocol": PROTOCOL,
        "experiment": "su4q-chain-n300-conformance",
        "items": ITEMS,
        "depth": depth,
        "values": ITEMS * 8,
        "atol": ATOL,
        "rtol": RTOL,
        "failure_count": failures,
        "nonfinite_value_count": nonfinite,
        "max_abs_error": float(np.nanmax(error)),
        "max_phase_aligned_abs_error": float(
            np.nanmax(np.abs(aligned - reference_complex))
        ),
        "max_state_norm_error": float(
            np.nanmax(np.abs(np.linalg.norm(output_complex, axis=1) - 1.0))
        ),
        "case_coverage": coverage,
        "expected_case_coverage": expected,
        "all_cases_represented": coverage == expected,
        "provenance_match": provenance_match,
        "transfer_contract_match": transfer_contract,
        "output_sha256": sha256_file(output_path),
        "performance_eligible": False,
        "host_only": host_only,
        "passed": (
            failures == 0
            and nonfinite == 0
            and coverage == expected
            and provenance_match
            and transfer_contract
        ),
    }
    target = session_dir / (
        "host_validation.json" if host_only else manifest["outputs"]["validation"]
    )
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
