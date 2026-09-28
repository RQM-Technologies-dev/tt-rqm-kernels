"""Fused multi-qubit SU4Q statevector conformance support."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from tt_rqm_kernels.su4q_conformance import (
    ATOL,
    BLOCK_CONVENTION,
    BLOCK_LANES,
    RQMC_COMMIT,
    RQME_COMMIT,
    RTOL,
    TT_METAL_COMMIT,
    TT_RQM_COMMIT,
    _compiler_cases,
    _require_base,
    _require_pin,
    _sentinel_cases,
    block_to_unitary,
    git_commit,
    routing_contract_checks,
    sha256_file,
)

PROTOCOL = "tt-rqm-su4q-statevector-conformance.v1"
METRICS_SCHEMA = "tt-rqm-su4q-statevector-conformance-metrics.v1"
VALIDATION_SCHEMA = "tt-rqm-su4q-statevector-conformance-validation.v1"
QUBITS = (3, 4, 6, 10, 15)
DEPTHS = (1, 8, 32, 128)
BATCH = 4


def quartet_indices(qubits: int, q0: int, q1: int) -> np.ndarray:
    if qubits not in QUBITS or q0 == q1 or min(q0, q1) < 0 or max(q0, q1) >= qubits:
        raise ValueError("invalid qubit count or target pair")
    values = []
    for compact in range(1 << (qubits - 2)):
        base = 0
        source = 0
        for bit in range(qubits):
            if bit in (q0, q1):
                continue
            base |= ((compact >> source) & 1) << bit
            source += 1
        values.append((base, base | (1 << q0), base | (1 << q1), base | (1 << q0) | (1 << q1)))
    return np.asarray(values, dtype=np.uint32)


def apply_pair(states: np.ndarray, matrix: np.ndarray, q0: int, q1: int, qubits: int) -> np.ndarray:
    output = np.asarray(states, dtype=np.complex128).copy()
    indices = quartet_indices(qubits, q0, q1)
    gathered = output[:, indices]
    transformed = np.einsum("rc,bkc->bkr", np.asarray(matrix, dtype=np.complex128), gathered)
    for lane in range(4):
        output[:, indices[:, lane]] = transformed[:, :, lane]
    return output


def _pair_class(q0: int, q1: int) -> str:
    if q0 > q1:
        return "reversed_order"
    if q1 - q0 == 1:
        return "adjacent"
    return "non_adjacent"


def _schedule(qubits: int, depth: int, case_count: int) -> list[dict[str, Any]]:
    if depth == 1:
        return [{"case_index": 5, "q0": 0, "q1": qubits - 1, "pair_class": "non_adjacent"}]
    schedule = []
    for layer in range(depth):
        slot = layer % 8
        if slot < 6:
            q0 = layer % (qubits - 1)
            q1 = q0 + 1
            if slot in (2, 5) and qubits > 3:
                q0, q1 = 0, qubits - 1
            case_index = slot
        elif slot == 6:
            q0, q1 = 0, qubits - 1
            case_index = 10
        else:
            q0, q1 = qubits - 1, 0
            case_index = 7
        if case_index >= case_count:
            raise ValueError("schedule references an unavailable case")
        schedule.append(
            {
                "case_index": case_index,
                "q0": q0,
                "q1": q1,
                "pair_class": _pair_class(q0, q1),
            }
        )
    return schedule


def _prove_placements(
    qubits: int,
    schedule: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    routing: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    placements = sorted(
        {
            (entry["case_index"], entry["q0"], entry["q1"])
            for entry in schedule
            if entry["case_index"] < 6
        }
    )
    evidence = []
    for case_index, q0, q1 in placements:
        case = cases[case_index]
        proof = routing[case_index]
        if proof.get("semantic_verified") is not True or not proof.get("selected_windows"):
            raise ValueError(f"pair-local proof missing for {case['id']}")
        if q0 == q1 or min(q0, q1) < 0 or max(q0, q1) >= qubits:
            raise ValueError("invalid placement pair")
        evidence.append(
            {
                "case_id": case["id"],
                "source_pair": [0, 1],
                "placed_pair": [q0, q1],
                "placement_contract": "pair_local_unitary_is_index_independent",
                "semantic_verified": True,
                "selected_windows": proof["selected_windows"],
                "rejected_windows": proof["rejected_windows"],
            }
        )
    return evidence


def prepare_session(
    output_dir: Path,
    *,
    qubits: int,
    depth: int,
    compiler_root: Path,
    entanglement_root: Path,
    tt_rqm_root: Path,
    tt_metal_root: Path,
) -> Path:
    if qubits not in QUBITS or depth not in DEPTHS:
        raise ValueError("unsupported qubit count or depth")
    for root, expected, label in (
        (compiler_root, RQMC_COMMIT, "rqm-compiler"),
        (entanglement_root, RQME_COMMIT, "rqm-entanglement"),
        (tt_metal_root, TT_METAL_COMMIT, "tt-metal"),
    ):
        _require_pin(root.resolve(), expected, label)
    _require_base(tt_rqm_root.resolve(), TT_RQM_COMMIT, "tt-rqm-kernels")
    output_dir.mkdir(parents=True, exist_ok=False)
    compiler_cases, routing = _compiler_cases()
    sentinels = _sentinel_cases()
    cases = compiler_cases + sentinels
    schedule = _schedule(qubits, depth, len(cases))
    placement_evidence = _prove_placements(qubits, schedule, cases, routing)
    amplitudes = 1 << qubits
    states = np.zeros((BATCH, amplitudes), dtype=np.complex128)
    states[0, 0] = 1
    states[1, -1] = 1
    states[2, 0] = states[2, -1] = 1 / np.sqrt(2)
    rng = np.random.default_rng(20260726 + 1000 * qubits + depth)
    random_state = rng.normal(size=amplitudes) + 1j * rng.normal(size=amplitudes)
    states[3] = random_state / np.linalg.norm(random_state)
    reference = states.copy()
    blocks = np.empty((depth, 20), dtype=np.float32)
    pairs = np.empty((depth, 2), dtype=np.uint32)
    case_indices = np.empty(depth, dtype=np.uint32)
    for layer, entry in enumerate(schedule):
        case = cases[entry["case_index"]]
        blocks[layer] = case["block"].astype(np.float32)
        pairs[layer] = (entry["q0"], entry["q1"])
        case_indices[layer] = entry["case_index"]
        reference = apply_pair(
            reference, case["unitary"], entry["q0"], entry["q1"], qubits
        )
    state_lanes = np.empty((BATCH, amplitudes, 2), dtype=np.float32)
    state_lanes[..., 0] = states.real.astype(np.float32)
    state_lanes[..., 1] = states.imag.astype(np.float32)
    reference_lanes = np.empty((BATCH, amplitudes, 2), dtype=np.float64)
    reference_lanes[..., 0] = reference.real
    reference_lanes[..., 1] = reference.imag
    paths = {
        "blocks": output_dir / "blocks.bin",
        "pairs": output_dir / "pairs.bin",
        "states": output_dir / "states.bin",
        "case_indices": output_dir / "case_indices.bin",
        "reference_states": output_dir / "reference_states.bin",
    }
    for name, value in (
        ("blocks", blocks),
        ("pairs", pairs),
        ("states", state_lanes),
        ("case_indices", case_indices),
        ("reference_states", reference_lanes),
    ):
        value.tofile(paths[name])
    used = sorted(set(int(value) for value in case_indices))
    manifest = {
        "schema": PROTOCOL,
        "experiment": "su4q-multiqubit-n300-conformance",
        "stage": "conformance",
        "performance_eligible": False,
        "dtype": "float32",
        "qubits": qubits,
        "depth": depth,
        "batch": BATCH,
        "block_shape": [depth, 20],
        "pair_shape": [depth, 2],
        "state_shape": [BATCH, amplitudes, 2],
        "block_lanes": list(BLOCK_LANES),
        "block_convention_version": BLOCK_CONVENTION,
        "basis_order": "little_endian",
        "quartet_order": "|q1 q0>",
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
                "id": cases[index]["id"],
                "kind": cases[index]["kind"],
                "oracle_source": (
                    "original_source_circuit_complex128"
                    if index < 6
                    else "explicit_convention_matrix"
                ),
            }
            for index in used
        ],
        "expected_case_indices": used,
        "expected_pair_classes": sorted(set(entry["pair_class"] for entry in schedule)),
        "schedule": schedule,
        "compiler_placement_evidence": placement_evidence,
        "routing_contract_checks": routing_contract_checks(),
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
    qubits, depth = int(manifest["qubits"]), int(manifest["depth"])
    amplitudes = 1 << qubits
    blocks = np.fromfile(session_dir / "blocks.bin", dtype=np.float32).reshape(depth, 20)
    pairs = np.fromfile(session_dir / "pairs.bin", dtype=np.uint32).reshape(depth, 2)
    lanes = np.fromfile(session_dir / "states.bin", dtype=np.float32).reshape(
        BATCH, amplitudes, 2
    )
    states = lanes[..., 0].astype(np.float64) + 1j * lanes[..., 1].astype(np.float64)
    for layer in range(depth):
        states = apply_pair(
            states,
            block_to_unitary(blocks[layer].astype(np.float64)),
            int(pairs[layer, 0]),
            int(pairs[layer, 1]),
            qubits,
        )
    output = np.empty((BATCH, amplitudes, 2), dtype=np.float32)
    output[..., 0], output[..., 1] = states.real, states.imag
    path = session_dir / "host_output_states.bin"
    output.tofile(path)
    return path


def validate_session(session_dir: Path, *, output_name: str | None = None) -> dict[str, Any]:
    manifest = json.loads((session_dir / "manifest.json").read_text())
    if manifest.get("schema") != PROTOCOL:
        raise ValueError("unsupported statevector protocol")
    qubits, depth = int(manifest["qubits"]), int(manifest["depth"])
    if qubits not in QUBITS or depth not in DEPTHS:
        raise ValueError("unsupported qubit count or depth")
    for record in manifest["inputs"].values():
        path = session_dir / record["file"]
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"input hash mismatch: {path.name}")
    amplitudes = 1 << qubits
    output_path = session_dir / (
        output_name if output_name else manifest["outputs"]["states"]
    )
    output = np.fromfile(output_path, dtype=np.float32)
    if output.size != BATCH * amplitudes * 2:
        raise ValueError("output size mismatch")
    output = output.reshape(BATCH, amplitudes, 2).astype(np.float64)
    reference = np.fromfile(
        session_dir / "reference_states.bin", dtype=np.float64
    ).reshape(BATCH, amplitudes, 2)
    error = np.abs(output - reference)
    threshold = ATOL + RTOL * np.abs(reference)
    nonfinite = int(np.count_nonzero(~np.isfinite(output)))
    failures = int(np.count_nonzero((error > threshold) | ~np.isfinite(output)))
    oc = output[..., 0] + 1j * output[..., 1]
    rc = reference[..., 0] + 1j * reference[..., 1]
    overlaps = np.sum(np.conj(oc) * rc, axis=1)
    phases = np.ones(BATCH, dtype=np.complex128)
    valid = np.abs(overlaps) > 0
    phases[valid] = overlaps[valid] / np.abs(overlaps[valid])
    aligned = oc * phases[:, None]
    cases = sorted(
        set(
            int(value)
            for value in np.fromfile(session_dir / "case_indices.bin", dtype=np.uint32)
        )
    )
    pair_classes = sorted(set(item["pair_class"] for item in manifest["schedule"]))
    host_only = output_name is not None
    provenance = transfer = True
    if not host_only:
        metrics = json.loads((session_dir / "metrics.json").read_text())
        metadata = metrics.get("candidate_metadata", {})
        provenance = (
            metrics.get("schema") == METRICS_SCHEMA
            and metrics.get("protocol") == PROTOCOL
            and metrics.get("qubits") == qubits
            and metrics.get("depth") == depth
            and metrics.get("performance_eligible") is False
            and metadata.get("tt_metal_commit") == TT_METAL_COMMIT
            and metadata.get("source_tree_clean") is True
            and metadata.get("device_id") == 0
            and metadata.get("device_count") == 1
            and metadata.get("device_create_count") == 1
            and metadata.get("device_close_count") == 1
            and metadata.get("program_count") == depth
            and metadata.get("dispatch_count") == depth
            and bool(metadata.get("candidate_sha256"))
            and bool(metadata.get("source_bundle_sha256"))
        )
        transfer = (
            metadata.get("initial_block_upload_count") == 1
            and metadata.get("initial_state_upload_count") == 1
            and metadata.get("intermediate_h2d_count") == 0
            and metadata.get("intermediate_d2h_count") == 0
            and metadata.get("final_state_download_count") == 1
        )
    result = {
        "schema": VALIDATION_SCHEMA,
        "protocol": PROTOCOL,
        "experiment": "su4q-multiqubit-n300-conformance",
        "qubits": qubits,
        "depth": depth,
        "batch": BATCH,
        "values": BATCH * amplitudes * 2,
        "failure_count": failures,
        "nonfinite_value_count": nonfinite,
        "max_abs_error": float(np.nanmax(error)),
        "max_phase_aligned_abs_error": float(np.nanmax(np.abs(aligned - rc))),
        "max_state_norm_error": float(
            np.nanmax(np.abs(np.linalg.norm(oc, axis=1) - 1.0))
        ),
        "case_coverage": cases,
        "all_cases_represented": cases == manifest["expected_case_indices"],
        "pair_class_coverage": pair_classes,
        "all_pair_classes_represented": pair_classes == manifest["expected_pair_classes"],
        "provenance_match": provenance,
        "transfer_contract_match": transfer,
        "output_sha256": sha256_file(output_path),
        "performance_eligible": False,
        "host_only": host_only,
        "passed": (
            failures == 0
            and nonfinite == 0
            and cases == manifest["expected_case_indices"]
            and pair_classes == manifest["expected_pair_classes"]
            and provenance
            and transfer
        ),
    }
    target = session_dir / (
        "host_validation.json" if host_only else manifest["outputs"]["validation"]
    )
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def source_bundle_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    package = root / "experimental/tt_metalium_su4q_statevector"
    for path in sorted(candidate for candidate in package.rglob("*") if candidate.is_file()):
        relative = path.relative_to(root).as_posix().encode()
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def git_clean(root: Path) -> bool:
    return not bool(
        subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
