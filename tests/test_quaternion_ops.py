from __future__ import annotations

import torch

from tt_rqm_kernels import qconj, qinverse, qmul, qnorm, qnormalize


def test_hamilton_product_identity() -> None:
    q = torch.randn(32, 4, dtype=torch.float64)
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)

    assert torch.allclose(qmul(identity, q), q)
    assert torch.allclose(qmul(q, identity), q)


def test_hamilton_product_basis_rules() -> None:
    one = torch.tensor([1.0, 0.0, 0.0, 0.0])
    i = torch.tensor([0.0, 1.0, 0.0, 0.0])
    j = torch.tensor([0.0, 0.0, 1.0, 0.0])
    k = torch.tensor([0.0, 0.0, 0.0, 1.0])

    assert torch.allclose(qmul(i, i), -one)
    assert torch.allclose(qmul(j, j), -one)
    assert torch.allclose(qmul(k, k), -one)
    assert torch.allclose(qmul(i, j), k)
    assert torch.allclose(qmul(j, k), i)
    assert torch.allclose(qmul(k, i), j)


def test_associativity_within_float_tolerance() -> None:
    a = torch.randn(16, 4, dtype=torch.float64)
    b = torch.randn(16, 4, dtype=torch.float64)
    c = torch.randn(16, 4, dtype=torch.float64)

    left = qmul(qmul(a, b), c)
    right = qmul(a, qmul(b, c))

    assert torch.allclose(left, right, rtol=1e-10, atol=1e-10)


def test_inverse_multiplies_to_identity() -> None:
    q = torch.randn(64, 4, dtype=torch.float64) + 0.1
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)

    left = qmul(q, qinverse(q))
    right = qmul(qinverse(q), q)

    assert torch.allclose(left, identity.expand_as(left), rtol=1e-10, atol=1e-10)
    assert torch.allclose(right, identity.expand_as(right), rtol=1e-10, atol=1e-10)


def test_conjugate_and_norm_relation() -> None:
    q = torch.randn(12, 4, dtype=torch.float64)
    product = qmul(q, qconj(q))
    expected_real = qnorm(q) ** 2

    assert torch.allclose(product[..., 0], expected_real)
    assert torch.allclose(product[..., 1:], torch.zeros_like(product[..., 1:]))


def test_normalize_rejects_zero_quaternion() -> None:
    zero = torch.zeros(4)

    try:
        qnormalize(zero)
    except ValueError as exc:
        assert "too close to zero" in str(exc)
    else:
        raise AssertionError("qnormalize should reject zero quaternions")
