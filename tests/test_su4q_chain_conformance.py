from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tt_rqm_kernels.su4q_chain_conformance import (
    DEPTHS,
    ITEMS,
    METRICS_SCHEMA,
    PROTOCOL,
    _case_for,
    host_candidate,
    validate_session,
)


def test_protocol_depths_and_item_count_are_frozen() -> None:
    assert PROTOCOL == "tt-rqm-su4q-chain-conformance.v1"
    assert DEPTHS == (1, 8, 32, 128)
    assert ITEMS == 128


def test_scenarios_include_order_inverse_and_accumulation() -> None:
    cases = [
        {"id": value}
        for value in (
            "c0", "c1", "c2", "c3", "c4", "c5",
            "sentinel_q1_tensor_q0_order",
            "sentinel_positive_cartan_sign",
            "chain_inverse_positive",
            "chain_inverse_negative",
            "chain_local_only",
            "sentinel_global_phase",
            "sentinel_identity",
            "sentinel_iswap_class",
        )
    ]
    assert _case_for(8, 0, cases)[1] == "noncommutative_ab"
    assert _case_for(9, 0, cases)[1] == "noncommutative_ba"
    assert _case_for(10, 0, cases)[1] == "inverse_cancellation"
    assert _case_for(11, 0, cases)[1] == "local_only"
    assert _case_for(12, 0, cases)[1] == "global_phase_accumulation"


@pytest.mark.parametrize("depth", (0, 2, 7, 129))
def test_validator_rejects_malformed_depth(tmp_path: Path, depth: int) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps({"schema": PROTOCOL, "depth": depth, "items": ITEMS})
    )
    with pytest.raises(ValueError, match="invalid chain depth"):
        validate_session(tmp_path)


def test_host_candidate_preserves_layer_order_and_global_phase(tmp_path: Path) -> None:
    identity = np.asarray((1.0, 0.0, 0.0, 0.0))
    block_a = np.asarray(
        (*identity, *identity, 0.0, 0.0, 0.0, *identity, *identity, 0.2),
        dtype=np.float32,
    )
    block_b = block_a.copy()
    block_b[19] = -0.05
    blocks = np.empty((8, ITEMS, 20), dtype=np.float32)
    blocks[0::2] = block_a
    blocks[1::2] = block_b
    states = np.zeros((ITEMS, 8), dtype=np.float32)
    states[:, 0] = 1.0
    reference = np.zeros((ITEMS, 8), dtype=np.float64)
    phase = 4 * (0.2 - 0.05)
    reference[:, 0] = np.cos(phase)
    reference[:, 1] = np.sin(phase)
    cases = np.zeros((8, ITEMS), dtype=np.uint32)
    scenarios = np.zeros(ITEMS, dtype=np.uint32)
    for name, values in (
        ("blocks", blocks),
        ("states", states),
        ("reference_states", reference),
        ("case_indices", cases),
        ("scenario_indices", scenarios),
    ):
        values.tofile(tmp_path / f"{name}.bin")
    import hashlib

    paths = {
        name: tmp_path / f"{name}.bin"
        for name in ("blocks", "states", "reference_states", "case_indices", "scenario_indices")
    }
    manifest = {
        "schema": PROTOCOL,
        "depth": 8,
        "items": ITEMS,
        "cases": [{"index": 0}],
        "inputs": {
            name: {
                "file": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in paths.items()
        },
        "outputs": {
            "states": "output_states.bin",
            "metrics": "metrics.json",
            "validation": "validation.json",
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    output = host_candidate(tmp_path)
    result = validate_session(tmp_path, output_name=output.name)
    assert result["passed"] is True
    assert result["max_abs_error"] < 1.0e-6


def test_device_validation_requires_transfer_contract(tmp_path: Path) -> None:
    states = np.zeros((ITEMS, 8), dtype=np.float32)
    reference = states.astype(np.float64)
    cases = np.zeros((ITEMS,), dtype=np.uint32)
    for name, values in (
        ("states", states),
        ("reference_states", reference),
        ("case_indices", cases),
    ):
        values.tofile(tmp_path / f"{name}.bin")
    import hashlib

    paths = {name: tmp_path / f"{name}.bin" for name in ("states", "reference_states", "case_indices")}
    manifest = {
        "schema": PROTOCOL,
        "depth": 1,
        "items": ITEMS,
        "cases": [{"index": 0}],
        "inputs": {
            name: {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for name, path in paths.items()
        },
        "outputs": {"states": "output_states.bin", "metrics": "metrics.json", "validation": "validation.json"},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    states.tofile(tmp_path / "output_states.bin")
    metrics = {
        "schema": METRICS_SCHEMA,
        "protocol": PROTOCOL,
        "performance_eligible": False,
        "depth": 1,
        "candidate_metadata": {
            "tt_metal_commit": "9802b80464cfd213b1146189753d8aedf86193fe",
            "device_id": 0,
            "device_count": 1,
            "device_create_count": 1,
            "device_close_count": 1,
            "program_count": 8,
            "source_tree_clean": True,
            "candidate_sha256": "x",
            "source_bundle_sha256": "y",
            "initial_block_upload_count": 1,
            "initial_state_upload_count": 1,
            "intermediate_h2d_count": 1,
            "intermediate_d2h_count": 0,
            "final_state_download_count": 1,
        },
    }
    (tmp_path / "metrics.json").write_text(json.dumps(metrics))
    result = validate_session(tmp_path)
    assert result["transfer_contract_match"] is False
    assert result["passed"] is False


def test_candidate_source_encodes_depth_major_offsets_and_single_transfers() -> None:
    root = Path("experimental/tt_metalium_su4q_conformance")
    source = (root / "src/su4q_conformance_candidate.cpp").read_text()
    reader = (root / "kernels/reader_stage.cpp").read_text()
    assert "layer * kBlockLanes * component_tiles" in source
    assert "for (uint32_t layer = 0; layer < depth; ++layer)" in source
    assert source.count("EnqueueWriteMeshBuffer") == 2
    assert source.count("EnqueueReadMeshBuffer") == 1
    assert "block_base_page +" in reader
