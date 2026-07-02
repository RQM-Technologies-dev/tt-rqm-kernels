from __future__ import annotations

import torch

from tt_rqm_kernels import (
    phase_difference,
    qmul,
    qnormalize,
    qrotate_vector,
    unwrap_phase,
    validate_qtensor,
)


def test_qmul_broadcasting_shape() -> None:
    a = torch.randn(2, 1, 4)
    b = torch.randn(3, 4)

    out = qmul(a, b)

    assert out.shape == (2, 3, 4)


def test_qrotate_broadcasting_shape() -> None:
    rotor = qnormalize(torch.randn(2, 1, 4))
    vector = torch.randn(3, 3)

    out = qrotate_vector(rotor, vector)

    assert out.shape == (2, 3, 3)


def test_validate_qtensor_rejects_wrong_final_dimension() -> None:
    try:
        validate_qtensor(torch.randn(3, 3))
    except ValueError as exc:
        assert "final dimension size 4" in str(exc)
    else:
        raise AssertionError("validate_qtensor should reject non-quaternion tensors")


def test_phase_difference_wraps_across_pi_boundary() -> None:
    current = torch.tensor([-3.10])
    previous = torch.tensor([3.10])

    delta = phase_difference(current, previous)

    assert torch.allclose(delta, torch.tensor([0.0831852]), atol=1e-6)


def test_unwrap_phase_sequence() -> None:
    wrapped = torch.tensor([3.00, 3.10, -3.08, -2.98])
    unwrapped = unwrap_phase(wrapped)

    assert torch.all(unwrapped[1:] > unwrapped[:-1])
    assert torch.allclose(unwrapped[:2], wrapped[:2])
