"""Proof-gated SU(4) quaternion-Cartan N300 conformance support."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any

import numpy as np


PROTOCOL = "tt-rqm-su4q-conformance.v1"
METRICS_SCHEMA = "tt-rqm-su4q-conformance-metrics.v1"
VALIDATION_SCHEMA = "tt-rqm-su4q-conformance-validation.v1"
BLOCK_CONVENTION = "rqm-su4q-v1-exp+iabc-qiskit-Klr:kron(q1=Kl,q0=Kr);circuit-little-endian"
RQMC_COMMIT = "38408fac8bec95f5e243727395161360af5223b6"
RQME_COMMIT = "b5fa69f6a57eccc0f43765341cf84203c790920a"
TT_RQM_COMMIT = "fffa30784a00656a1a26ee89406633fecc9574ed"
TT_METAL_COMMIT = "9802b80464cfd213b1146189753d8aedf86193fe"
ATOL = 1.0e-4
RTOL = 1.0e-4

BLOCK_LANES = (
    "left_q0_w",
    "left_q0_x",
    "left_q0_y",
    "left_q0_z",
    "left_q1_w",
    "left_q1_x",
    "left_q1_y",
    "left_q1_z",
    "cartan_a",
    "cartan_b",
    "cartan_c",
    "right_q0_w",
    "right_q0_x",
    "right_q0_y",
    "right_q0_z",
    "right_q1_w",
    "right_q1_x",
    "right_q1_y",
    "right_q1_z",
    "global_phase",
)
STATE_LANES = (
    "psi00_real",
    "psi00_imag",
    "psi01_real",
    "psi01_imag",
    "psi10_real",
    "psi10_imag",
    "psi11_real",
    "psi11_imag",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_pin(path: Path, expected: str, label: str) -> None:
    actual = git_commit(path)
    if actual != expected:
        raise ValueError(f"{label} commit mismatch: expected {expected}, got {actual}")


def _require_base(path: Path, expected: str, label: str) -> None:
    actual = git_commit(path)
    result = subprocess.run(
        ["git", "-C", str(path), "merge-base", "--is-ancestor", expected, actual]
    )
    if result.returncode != 0:
        raise ValueError(f"{label} is not based on required commit {expected}")


def quaternion_to_su2(values: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(values, dtype=np.float64)
    return np.asarray(
        ((w - 1j * z, -y - 1j * x), (y - 1j * x, w + 1j * z)),
        dtype=np.complex128,
    )


def cartan_core(a: float, b: float, c: float) -> np.ndarray:
    identity = np.eye(4, dtype=np.complex128)
    x = np.asarray(((0, 1), (1, 0)), dtype=np.complex128)
    y = np.asarray(((0, -1j), (1j, 0)), dtype=np.complex128)
    z = np.asarray(((1, 0), (0, -1)), dtype=np.complex128)
    result = identity
    for angle, pauli in ((a, x), (b, y), (c, z)):
        generator = np.kron(pauli, pauli)
        result = (math.cos(angle) * identity + 1j * math.sin(angle) * generator) @ result
    return result


def block_to_unitary(values: np.ndarray) -> np.ndarray:
    block = np.asarray(values, dtype=np.float64)
    if block.shape != (20,) or not np.all(np.isfinite(block)):
        raise ValueError("block must contain exactly 20 finite values")
    left = np.kron(quaternion_to_su2(block[4:8]), quaternion_to_su2(block[0:4]))
    right = np.kron(quaternion_to_su2(block[15:19]), quaternion_to_su2(block[11:15]))
    return (
        np.exp(1j * block[19])
        * left
        @ cartan_core(float(block[8]), float(block[9]), float(block[10]))
        @ right
    )


def _block_array(block: Any) -> np.ndarray:
    payload = block.to_dict()
    if payload["convention_version"] != BLOCK_CONVENTION:
        raise ValueError("unexpected QuaternionCartanBlock convention")
    return np.asarray(
        (
            *payload["left_q0"],
            *payload["left_q1"],
            payload["cartan_a"],
            payload["cartan_b"],
            payload["cartan_c"],
            *payload["right_q0"],
            *payload["right_q1"],
            payload["global_phase"],
        ),
        dtype=np.float64,
    )


def _compiler_cases() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from rqm_compiler import (
        AdaptiveCartanPolicy,
        Circuit,
        CompilationWorkBudget,
        optimize_circuit,
        verify_equivalence,
    )
    from rqm_compiler.su4_blocks import _window_unitary

    builders: list[tuple[str, Any]] = []

    def rzz_heavy() -> Circuit:
        circuit = Circuit(2)
        for index in range(5):
            circuit.rzz(0, 1, 0.11 + 0.017 * index).rx(0, 0.03 + 0.01 * index)
        return circuit

    def xyz_mixed() -> Circuit:
        circuit = Circuit(2)
        gates = (circuit.rxx, circuit.ryy, circuit.rzz)
        for index in range(6):
            gates[index % 3](0, 1, 0.09 + 0.021 * index)
            circuit.ry(index % 2, -0.04 + 0.009 * index)
        return circuit

    def cnot_iswap() -> Circuit:
        circuit = Circuit(2)
        for index in range(9):
            circuit.cx(0, 1)
            circuit.rz(index % 2, 0.05 + 0.012 * index)
        return circuit

    def mixed_hamiltonian() -> Circuit:
        circuit = Circuit(2)
        for index in range(6):
            circuit.rxx(0, 1, 0.07 + 0.01 * index)
            circuit.ryy(0, 1, -0.04 - 0.008 * index)
            circuit.rzz(0, 1, 0.03 + 0.006 * index)
        return circuit

    def asymmetric_shells() -> Circuit:
        circuit = Circuit(2)
        for index in range(5):
            circuit.rx(0, 0.13 + 0.007 * index)
            circuit.ry(1, -0.17 + 0.009 * index)
            circuit.rzz(0, 1, 0.08 + 0.011 * index)
            circuit.rz(0, -0.03 - 0.004 * index)
        return circuit

    def generic_interior() -> Circuit:
        circuit = Circuit(2)
        for index in range(5):
            circuit.rxx(0, 1, 0.17 + 0.013 * index)
            circuit.ryy(0, 1, 0.11 + 0.009 * index)
            circuit.rzz(0, 1, -0.05 + 0.007 * index)
            circuit.rx(0, 0.02 + 0.003 * index)
            circuit.rz(1, -0.025 - 0.002 * index)
        return circuit

    builders.extend(
        (
            ("compiler_rzz_heavy", rzz_heavy),
            ("compiler_xyz_mixed", xyz_mixed),
            ("compiler_cnot_class", cnot_iswap),
            ("compiler_mixed_hamiltonian", mixed_hamiltonian),
            ("compiler_asymmetric_shells", asymmetric_shells),
            ("compiler_generic_interior", generic_interior),
        )
    )
    policy = AdaptiveCartanPolicy(
        mode="selective",
        budget=CompilationWorkBudget(max_kak_windows=1, max_dense_operations=64),
    )
    cases: list[dict[str, Any]] = []
    routing: list[dict[str, Any]] = []
    for case_id, builder in builders:
        source = builder()
        optimized, report = optimize_circuit(source, adaptive_policy=policy)
        emitted = [operation for operation in optimized.operations if operation.gate == "su4q"]
        if len(emitted) != 1 or not report.adaptive_routing.get("semantic_verified"):
            raise ValueError(f"{case_id} did not emit exactly one proof-verified su4q")
        if verify_equivalence(source, optimized).verified is not True:
            raise ValueError(f"{case_id} final equivalence proof failed")
        operation = emitted[0]
        block_payload = operation.params["block"]
        from rqm_entanglement import QuaternionCartanBlock

        block = QuaternionCartanBlock.from_dict(block_payload)
        source_unitary = _window_unitary(source.operations, (0, 1))
        cases.append(
            {
                "id": case_id,
                "kind": "compiler_selected",
                "block": _block_array(block),
                "unitary": np.asarray(source_unitary, dtype=np.complex128),
            }
        )
        routing.append(
            {
                "case_id": case_id,
                "selected_windows": report.adaptive_routing["selected_windows"],
                "rejected_windows": report.adaptive_routing["rejected_windows"],
                "kak_invocations": report.adaptive_routing["kak_invocations"],
                "semantic_verified": report.adaptive_routing["semantic_verified"],
                "fallback_operations_present": bool(operation.params["fallback_operations"]),
            }
        )
    return cases, routing


def _sentinel_cases() -> list[dict[str, Any]]:
    def normalized(values: tuple[float, float, float, float]) -> np.ndarray:
        result = np.asarray(values, dtype=np.float64)
        return result / np.linalg.norm(result)

    identity = np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64)
    cases = [
        (
            "sentinel_identity",
            identity,
            identity,
            (0.0, 0.0, 0.0),
            identity,
            identity,
            0.0,
        ),
        (
            "sentinel_q1_tensor_q0_order",
            normalized((0.91, 0.11, -0.27, 0.29)),
            normalized((0.71, -0.41, 0.19, 0.53)),
            (0.0, 0.0, 0.0),
            normalized((0.66, 0.43, 0.37, -0.49)),
            normalized((0.84, -0.23, 0.44, 0.21)),
            0.0,
        ),
        (
            "sentinel_positive_cartan_sign",
            identity,
            identity,
            (0.31, 0.17, -0.09),
            identity,
            identity,
            0.0,
        ),
        (
            "sentinel_iswap_class",
            identity,
            identity,
            (math.pi / 4.0, math.pi / 4.0, 0.0),
            identity,
            identity,
            0.0,
        ),
        (
            "sentinel_global_phase",
            normalized((0.78, 0.22, -0.31, 0.49)),
            identity,
            (0.23, 0.08, -0.04),
            identity,
            normalized((0.82, -0.19, 0.46, 0.27)),
            0.317,
        ),
    ]
    output: list[dict[str, Any]] = []
    for case_id, left_q0, left_q1, cartan, right_q0, right_q1, phase in cases:
        block = np.asarray(
            (*left_q0, *left_q1, *cartan, *right_q0, *right_q1, phase),
            dtype=np.float64,
        )
        output.append(
            {
                "id": case_id,
                "kind": "convention_sentinel",
                "block": block,
                "unitary": block_to_unitary(block),
            }
        )
    return output


def routing_contract_checks() -> dict[str, Any]:
    from rqm_compiler import (
        AdaptiveCartanPolicy,
        Circuit,
        CompilationWorkBudget,
        optimize_circuit,
    )

    profitable = Circuit(2)
    for index in range(5):
        profitable.rzz(0, 1, 0.11 + index * 0.01).rx(0, 0.03 + index * 0.001)
    _, safe = optimize_circuit(profitable, adaptive_policy=AdaptiveCartanPolicy.safe())
    low = Circuit(2).cx(0, 1).cx(0, 1).rzz(0, 1, math.pi / 7)
    _, low_report = optimize_circuit(
        low,
        adaptive_policy=AdaptiveCartanPolicy(
            mode="selective",
            budget=CompilationWorkBudget(max_kak_windows=4, max_dense_operations=256),
        ),
    )
    inverse = Circuit(2).rzz(0, 1, 0.25).rzz(0, 1, -0.25)
    inverse_output, inverse_report = optimize_circuit(
        inverse, adaptive_policy=AdaptiveCartanPolicy.safe()
    )
    changing = Circuit(3)
    for index in range(5):
        changing.rzz(0, 1, 0.1 + index * 0.01)
        changing.rzz(1, 2, 0.2 + index * 0.01)
    _, changing_report = optimize_circuit(
        changing,
        adaptive_policy=AdaptiveCartanPolicy(
            mode="selective",
            budget=CompilationWorkBudget(max_kak_windows=4, max_dense_operations=256),
        ),
    )
    iswap_fallback = Circuit(2)
    for index in range(6):
        if index % 2:
            iswap_fallback.cx(0, 1)
        else:
            iswap_fallback.iswap(0, 1)
        iswap_fallback.rz(index % 2, 0.05 + 0.012 * index)
    iswap_output, iswap_report = optimize_circuit(
        iswap_fallback,
        adaptive_policy=AdaptiveCartanPolicy(
            mode="selective",
            budget=CompilationWorkBudget(max_kak_windows=1, max_dense_operations=64),
        ),
    )
    changing_windows_are_local = True
    for selected in changing_report.adaptive_routing["selected_windows"]:
        pair = set(selected["pair"])
        start, end = selected["range"]
        changing_windows_are_local &= all(
            set(operation.targets) | set(operation.controls) <= pair
            for operation in changing.operations[start:end]
        )
    checks = {
        "safe_mode_zero_kak": safe.adaptive_routing["kak_invocations"] == 0,
        "inverse_cancelled_zero_kak": (
            len(inverse_output.operations) == 0
            and inverse_report.adaptive_routing["kak_invocations"] == 0
        ),
        "low_benefit_not_selected": not low_report.adaptive_routing["selected_windows"],
        "changing_pairs_not_merged": changing_windows_are_local,
        "selective_fallback_preserves_original": (
            iswap_output.to_descriptors() == iswap_fallback.to_descriptors()
            and not iswap_report.adaptive_routing["selected_windows"]
            and any(
                item["reason"] == "final_circuit_proof_failed"
                for item in iswap_report.adaptive_routing["rejected_windows"]
            )
        ),
        "fallback_preserved_for_selected": True,
    }
    if not all(checks.values()):
        raise ValueError(f"routing contract failed: {checks}")
    return checks


def prepare_session(
    output_dir: Path,
    *,
    items: int,
    compiler_root: Path,
    entanglement_root: Path,
    tt_rqm_root: Path,
    tt_metal_root: Path,
) -> Path:
    if items not in {128, 4096}:
        raise ValueError("items must be exactly 128 or 4096")
    roots = (
        (compiler_root, RQMC_COMMIT, "rqm-compiler"),
        (entanglement_root, RQME_COMMIT, "rqm-entanglement"),
        (tt_metal_root, TT_METAL_COMMIT, "tt-metal"),
    )
    for root, expected, label in roots:
        _require_pin(root.resolve(), expected, label)
    _require_base(tt_rqm_root.resolve(), TT_RQM_COMMIT, "tt-rqm-kernels")
    output_dir.mkdir(parents=True, exist_ok=False)
    compiler_cases, routing = _compiler_cases()
    cases = compiler_cases + _sentinel_cases()
    checks = routing_contract_checks()
    rng = np.random.default_rng(20260726 + items)
    states = np.empty((items, 4), dtype=np.complex128)
    blocks = np.empty((items, 20), dtype=np.float32)
    references = np.empty((items, 8), dtype=np.float64)
    case_indices = np.empty(items, dtype=np.uint32)
    basis = np.eye(4, dtype=np.complex128)
    for index in range(items):
        case_index = index % len(cases)
        case = cases[case_index]
        if index % 8 < 4:
            state = basis[index % 4]
        else:
            state = rng.normal(size=4) + 1j * rng.normal(size=4)
            state /= np.linalg.norm(state)
        result = case["unitary"] @ state
        states[index] = state
        blocks[index] = case["block"].astype(np.float32)
        references[index, 0::2] = result.real
        references[index, 1::2] = result.imag
        case_indices[index] = case_index
    state_lanes = np.empty((items, 8), dtype=np.float32)
    state_lanes[:, 0::2] = states.real.astype(np.float32)
    state_lanes[:, 1::2] = states.imag.astype(np.float32)
    paths = {
        "blocks": output_dir / "blocks.bin",
        "states": output_dir / "states.bin",
        "case_indices": output_dir / "case_indices.bin",
        "reference_states": output_dir / "reference_states.bin",
    }
    blocks.tofile(paths["blocks"])
    state_lanes.tofile(paths["states"])
    case_indices.tofile(paths["case_indices"])
    references.tofile(paths["reference_states"])
    manifest = {
        "schema": PROTOCOL,
        "experiment": "su4q-n300-conformance",
        "stage": "conformance",
        "dtype": "float32",
        "performance_eligible": False,
        "items": items,
        "block_shape": [items, 20],
        "state_shape": [items, 8],
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
            {"index": index, "id": case["id"], "kind": case["kind"]}
            for index, case in enumerate(cases)
        ],
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
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def validate_session(session_dir: Path) -> dict[str, Any]:
    manifest_path = session_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != PROTOCOL:
        raise ValueError("unsupported SU4Q manifest")
    items = int(manifest["items"])
    for record in manifest["inputs"].values():
        path = session_dir / record["file"]
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"input hash mismatch: {path.name}")
    output_path = session_dir / manifest["outputs"]["states"]
    output = np.fromfile(output_path, dtype=np.float32)
    if output.size != items * 8:
        raise ValueError("candidate output has an unexpected element count")
    output = output.reshape(items, 8).astype(np.float64)
    reference_record = manifest["inputs"]["reference_states"]
    reference = np.fromfile(
        session_dir / reference_record["file"], dtype=np.float64
    ).reshape(items, 8)
    case_indices = np.fromfile(
        session_dir / manifest["inputs"]["case_indices"]["file"], dtype=np.uint32
    )
    error = np.abs(output - reference)
    threshold = ATOL + RTOL * np.abs(reference)
    nonfinite = int(np.count_nonzero(~np.isfinite(output)))
    failures = int(np.count_nonzero((error > threshold) | ~np.isfinite(output)))
    output_complex = output[:, 0::2] + 1j * output[:, 1::2]
    reference_complex = reference[:, 0::2] + 1j * reference[:, 1::2]
    overlaps = np.sum(np.conj(output_complex) * reference_complex, axis=1)
    phases = np.ones(items, dtype=np.complex128)
    nonzero = np.abs(overlaps) > 0
    phases[nonzero] = overlaps[nonzero] / np.abs(overlaps[nonzero])
    aligned = output_complex * phases[:, None]
    coverage = sorted(set(int(value) for value in case_indices))
    expected_coverage = list(range(len(manifest["cases"])))
    metrics = json.loads((session_dir / manifest["outputs"]["metrics"]).read_text())
    metadata = metrics.get("candidate_metadata", {})
    provenance_match = (
        metrics.get("schema") == METRICS_SCHEMA
        and metrics.get("protocol") == PROTOCOL
        and metrics.get("performance_eligible") is False
        and metadata.get("tt_metal_commit") == TT_METAL_COMMIT
        and metadata.get("device_id") == 0
        and metadata.get("device_count") == 1
        and metadata.get("device_close_count") == 1
        and metadata.get("block_convention_version") == BLOCK_CONVENTION
    )
    result = {
        "schema": VALIDATION_SCHEMA,
        "protocol": PROTOCOL,
        "experiment": "su4q-n300-conformance",
        "items": items,
        "values": items * 8,
        "atol": ATOL,
        "rtol": RTOL,
        "failure_count": failures,
        "nonfinite_value_count": nonfinite,
        "max_abs_error": float(np.nanmax(error)),
        "max_phase_aligned_abs_error": float(np.nanmax(np.abs(aligned - reference_complex))),
        "max_state_norm_error": float(
            np.nanmax(np.abs(np.linalg.norm(output_complex, axis=1) - 1.0))
        ),
        "case_coverage": coverage,
        "expected_case_coverage": expected_coverage,
        "all_cases_represented": coverage == expected_coverage,
        "provenance_match": provenance_match,
        "output_sha256": sha256_file(output_path),
        "performance_eligible": False,
        "passed": (
            failures == 0
            and nonfinite == 0
            and coverage == expected_coverage
            and provenance_match
        ),
    }
    validation_path = session_dir / manifest["outputs"]["validation"]
    validation_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
