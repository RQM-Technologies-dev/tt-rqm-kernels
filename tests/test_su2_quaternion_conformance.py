"""Independent SU(2) conformance checks for the public quaternion contract.

The test-only matrix oracle reproduces the canonical ``rqm-core`` convention:

``U([w, x, y, z]) = [[w - i z, -y - i x], [y - i x, w + i z]]``.

For the Hamilton-product order used by :func:`tt_rqm_kernels.qmul`, this is a
homomorphism: ``U(qmul(a, b)) = U(a) @ U(b)``. The oracle intentionally uses
ordinary complex matrices rather than the production quaternion operations.
"""

from __future__ import annotations

import math

import pytest
import torch

from tt_rqm_kernels import qconj, qinverse, qmul, qnormalize


def _su2_from_unit_quaternion(q: torch.Tensor) -> torch.Tensor:
    """Map unit ``[w, x, y, z]`` quaternions to canonical complex SU(2)."""

    w, x, y, z = q.unbind(dim=-1)
    top = torch.stack((torch.complex(w, -z), torch.complex(-y, -x)), dim=-1)
    bottom = torch.stack((torch.complex(y, -x), torch.complex(w, z)), dim=-1)
    return torch.stack((top, bottom), dim=-2)


def _unit_batch(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260712)
    a = qnormalize(torch.randn(32, 4, generator=generator, dtype=dtype))
    b = qnormalize(torch.randn(32, 4, generator=generator, dtype=dtype))
    return a, b


def _tolerance(dtype: torch.dtype) -> float:
    return 1e-11 if dtype == torch.float64 else 3e-6


def test_canonical_su2_mapping_covers_identity_and_quaternion_basis() -> None:
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    i = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float64)
    j = torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float64)
    k = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64)

    assert torch.allclose(_su2_from_unit_quaternion(identity), torch.eye(2, dtype=torch.complex128))
    assert torch.allclose(
        _su2_from_unit_quaternion(i),
        torch.tensor([[0.0j, -1.0j], [-1.0j, 0.0j]], dtype=torch.complex128),
    )
    assert torch.allclose(
        _su2_from_unit_quaternion(j),
        torch.tensor([[0.0j, -1.0], [1.0, 0.0j]], dtype=torch.complex128),
    )
    assert torch.allclose(
        _su2_from_unit_quaternion(k),
        torch.tensor([[-1.0j, 0.0j], [0.0j, 1.0j]], dtype=torch.complex128),
    )


def test_canonical_su2_mapping_matches_x_y_and_z_axis_rotations() -> None:
    half_angle = math.pi / 6.0
    c = math.cos(half_angle)
    s = math.sin(half_angle)
    rotors_and_expected = (
        (
            torch.tensor([c, s, 0.0, 0.0], dtype=torch.float64),
            torch.tensor([[c, -1j * s], [-1j * s, c]], dtype=torch.complex128),
        ),
        (
            torch.tensor([c, 0.0, s, 0.0], dtype=torch.float64),
            torch.tensor([[c, -s], [s, c]], dtype=torch.complex128),
        ),
        (
            torch.tensor([c, 0.0, 0.0, s], dtype=torch.float64),
            torch.tensor([[complex(c, -s), 0.0j], [0.0j, complex(c, s)]], dtype=torch.complex128),
        ),
    )

    for rotor, expected in rotors_and_expected:
        assert torch.allclose(_su2_from_unit_quaternion(rotor), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_normalized_quaternions_map_to_unit_determinant_matrices(dtype: torch.dtype) -> None:
    a, _ = _unit_batch(dtype)
    matrices = _su2_from_unit_quaternion(a)
    identity = torch.eye(2, dtype=matrices.dtype).expand_as(matrices)
    tolerance = _tolerance(dtype)

    assert torch.allclose(
        matrices @ matrices.transpose(-2, -1).conj(), identity, rtol=tolerance, atol=tolerance
    )
    assert torch.allclose(
        torch.linalg.det(matrices), torch.ones(32, dtype=matrices.dtype), rtol=tolerance, atol=tolerance
    )


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_su2_oracle_preserves_qmul_order_conjugation_and_inverse(dtype: torch.dtype) -> None:
    a, b = _unit_batch(dtype)
    tolerance = _tolerance(dtype)
    ua = _su2_from_unit_quaternion(a)
    ub = _su2_from_unit_quaternion(b)

    assert torch.allclose(
        _su2_from_unit_quaternion(qmul(a, b)), ua @ ub, rtol=tolerance, atol=tolerance
    )
    assert torch.allclose(
        _su2_from_unit_quaternion(qconj(a)), ua.transpose(-2, -1).conj(), rtol=tolerance, atol=tolerance
    )
    assert torch.allclose(
        _su2_from_unit_quaternion(qinverse(a)), torch.linalg.inv(ua), rtol=tolerance, atol=tolerance
    )


def test_local_kron_composition_matches_pairwise_quaternion_composition() -> None:
    a, b = _unit_batch(torch.float64)
    c, d = b[:8], a[:8]
    left = torch.kron(_su2_from_unit_quaternion(a[0]), _su2_from_unit_quaternion(b[0]))
    right = torch.kron(_su2_from_unit_quaternion(c[0]), _su2_from_unit_quaternion(d[0]))
    expected = torch.kron(
        _su2_from_unit_quaternion(qmul(a[0], c[0])),
        _su2_from_unit_quaternion(qmul(b[0], d[0])),
    )

    assert torch.allclose(left @ right, expected, rtol=1e-12, atol=1e-12)
