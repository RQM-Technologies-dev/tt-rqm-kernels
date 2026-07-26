from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tt_rqm_kernels.su4q_conformance import (
    ATOL,
    BLOCK_CONVENTION,
    METRICS_SCHEMA,
    PROTOCOL,
    RTOL,
    TT_METAL_COMMIT,
    block_to_unitary,
    cartan_core,
    quaternion_to_su2,
    validate_session,
)


def _normalized(values: tuple[float, float, float, float]) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    return result / np.linalg.norm(result)


def test_quaternion_matrix_and_factor_order_are_explicit() -> None:
    q0 = _normalized((0.91, 0.11, -0.27, 0.29))
    q1 = _normalized((0.71, -0.41, 0.19, 0.53))
    identity = (1.0, 0.0, 0.0, 0.0)
    block = np.asarray((*q0, *q1, 0.0, 0.0, 0.0, *identity, *identity, 0.0))
    actual = block_to_unitary(block)
    correct = np.kron(quaternion_to_su2(q1), quaternion_to_su2(q0))
    swapped = np.kron(quaternion_to_su2(q0), quaternion_to_su2(q1))
    assert np.allclose(actual, correct, atol=1e-14, rtol=0.0)
    assert np.max(np.abs(actual - swapped)) > 1e-3


def test_cartan_uses_positive_exponent_sign() -> None:
    angle = 0.31
    x = np.asarray(((0, 1), (1, 0)), dtype=np.complex128)
    xx = np.kron(x, x)
    expected = math.cos(angle) * np.eye(4) + 1j * math.sin(angle) * xx
    wrong_sign = math.cos(angle) * np.eye(4) - 1j * math.sin(angle) * xx
    actual = cartan_core(angle, 0.0, 0.0)
    assert np.allclose(actual, expected, atol=1e-14, rtol=0.0)
    assert np.max(np.abs(actual - wrong_sign)) > 0.1


def test_block_global_phase_is_not_discarded() -> None:
    identity = (1.0, 0.0, 0.0, 0.0)
    phase = 0.317
    block = np.asarray((*identity, *identity, 0.0, 0.0, 0.0, *identity, *identity, phase))
    assert np.allclose(block_to_unitary(block), np.exp(1j * phase) * np.eye(4))


@pytest.mark.parametrize(
    "bad",
    (np.zeros(19), np.zeros(21), np.full(20, np.nan)),
)
def test_block_rejects_malformed_values(bad: np.ndarray) -> None:
    with pytest.raises(ValueError):
        block_to_unitary(bad)


def test_validation_checks_all_values_cases_and_provenance(tmp_path: Path) -> None:
    items = 2
    blocks = np.zeros((items, 20), dtype=np.float32)
    states = np.zeros((items, 8), dtype=np.float32)
    cases = np.asarray((0, 1), dtype=np.uint32)
    reference = np.asarray(
        ((1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),) * items,
        dtype=np.float64,
    )
    paths = {}
    for name, values in (
        ("blocks", blocks),
        ("states", states),
        ("case_indices", cases),
        ("reference_states", reference),
    ):
        path = tmp_path / f"{name}.bin"
        values.tofile(path)
        paths[name] = path
    output = reference.astype(np.float32)
    output.tofile(tmp_path / "output_states.bin")
    import hashlib

    manifest = {
        "schema": PROTOCOL,
        "items": items,
        "cases": [{"index": 0}, {"index": 1}],
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
    metrics = {
        "schema": METRICS_SCHEMA,
        "protocol": PROTOCOL,
        "performance_eligible": False,
        "candidate_metadata": {
            "tt_metal_commit": TT_METAL_COMMIT,
            "device_id": 0,
            "device_count": 1,
            "device_close_count": 1,
            "block_convention_version": BLOCK_CONVENTION,
        },
    }
    (tmp_path / "metrics.json").write_text(json.dumps(metrics))
    result = validate_session(tmp_path)
    assert result["passed"] is True
    assert result["failure_count"] == 0
    assert result["nonfinite_value_count"] == 0
    assert result["max_abs_error"] <= ATOL + RTOL
