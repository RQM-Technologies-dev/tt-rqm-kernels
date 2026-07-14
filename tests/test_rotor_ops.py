from __future__ import annotations

import torch

from tt_rqm_kernels import (
    qconj,
    qmul,
    qnormalize,
    qrotate_vector,
    validate_unit_rotor,
    vector_to_pure_quaternion,
)


def test_vector_to_pure_quaternion() -> None:
    vectors = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0]])
    pure = vector_to_pure_quaternion(vectors)

    assert pure.shape == (2, 4)
    assert torch.allclose(pure[..., 0], torch.zeros(2))
    assert torch.allclose(pure[..., 1:], vectors)


def test_unit_rotor_validation() -> None:
    rotor = qnormalize(torch.tensor([0.5, 1.0, 0.0, 0.0]))
    assert validate_unit_rotor(rotor) is rotor


def test_unit_quaternion_rotation_preserves_vector_norm() -> None:
    rotors = qnormalize(torch.randn(128, 4, dtype=torch.float64))
    vectors = torch.randn(128, 3, dtype=torch.float64)

    rotated = qrotate_vector(rotors, vectors)

    assert torch.allclose(
        torch.linalg.vector_norm(rotated, dim=-1),
        torch.linalg.vector_norm(vectors, dim=-1),
        rtol=1e-10,
        atol=1e-10,
    )


def test_known_z_axis_rotation() -> None:
    angle = torch.tensor(torch.pi / 2.0)
    rotor = torch.tensor([torch.cos(angle / 2.0), 0.0, 0.0, torch.sin(angle / 2.0)])
    vector = torch.tensor([1.0, 0.0, 0.0])

    rotated = qrotate_vector(rotor, vector)

    assert torch.allclose(rotated, torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)


def test_rotor_product_applies_rightmost_rotation_first() -> None:
    angle = torch.tensor(torch.pi / 2.0, dtype=torch.float64)
    r1 = torch.tensor(
        [torch.cos(angle / 2.0), torch.sin(angle / 2.0), 0.0, 0.0], dtype=torch.float64
    )
    r2 = torch.tensor(
        [torch.cos(angle / 2.0), 0.0, torch.sin(angle / 2.0), 0.0], dtype=torch.float64
    )
    vector = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)

    composed = qrotate_vector(qmul(r2, r1), vector)
    applied_in_sequence = qrotate_vector(r2, qrotate_vector(r1, vector))

    assert torch.allclose(composed, applied_in_sequence, rtol=1e-12, atol=1e-12)


def test_quaternion_double_cover_preserves_vector_rotation() -> None:
    rotor = qnormalize(torch.tensor([1.0, -2.0, 3.0, -4.0], dtype=torch.float64))
    vectors = torch.tensor(
        [[1.0, 2.0, 3.0], [-2.0, 0.5, 4.0]], dtype=torch.float64
    )

    assert torch.allclose(
        qrotate_vector(rotor, vectors),
        qrotate_vector(-rotor, vectors),
        rtol=1e-12,
        atol=1e-12,
    )


def test_pure_quaternion_rotation_has_zero_real_lane() -> None:
    rotor = qnormalize(torch.tensor([1.0, -2.0, 3.0, -4.0], dtype=torch.float64))
    vectors = torch.tensor(
        [[1.0, 2.0, 3.0], [-2.0, 0.5, 4.0]], dtype=torch.float64
    )
    pure_vectors = vector_to_pure_quaternion(vectors)
    rotated_pure = qmul(qmul(rotor, pure_vectors), qconj(rotor))

    assert torch.allclose(
        rotated_pure[..., 0], torch.zeros_like(rotated_pure[..., 0]), rtol=1e-12, atol=1e-12
    )
    assert torch.allclose(
        rotated_pure[..., 1:], qrotate_vector(rotor, vectors), rtol=1e-12, atol=1e-12
    )
