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
    assert torch.allclose(qmul(j, i), -k)
    assert torch.allclose(qmul(k, j), -i)
    assert torch.allclose(qmul(i, k), -j)


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


def test_norm_is_multiplicative_for_deterministic_batches() -> None:
    a = torch.tensor(
        [[1.0, -2.0, 3.0, -4.0], [0.5, 1.5, -2.5, 3.5], [-1.0, 0.0, 2.0, 1.0]],
        dtype=torch.float64,
    )
    b = torch.tensor(
        [[-3.0, 2.0, 1.0, -0.5], [2.0, -1.0, 0.25, 0.75], [1.0, -4.0, 0.5, 2.0]],
        dtype=torch.float64,
    )

    assert torch.allclose(
        qnorm(qmul(a, b)),
        qnorm(a) * qnorm(b),
        rtol=1e-12,
        atol=1e-12,
    )


def test_conjugation_reverses_hamilton_product_order() -> None:
    a = torch.tensor(
        [[1.0, -2.0, 3.0, -4.0], [0.5, 1.5, -2.5, 3.5]], dtype=torch.float64
    )
    b = torch.tensor(
        [[-3.0, 2.0, 1.0, -0.5], [2.0, -1.0, 0.25, 0.75]], dtype=torch.float64
    )

    assert torch.allclose(
        qconj(qmul(a, b)),
        qmul(qconj(b), qconj(a)),
        rtol=1e-12,
        atol=1e-12,
    )


def test_unit_rotors_are_closed_under_hamilton_product() -> None:
    a = qnormalize(
        torch.tensor([[1.0, -2.0, 3.0, -4.0], [0.5, 1.5, -2.5, 3.5]], dtype=torch.float64)
    )
    b = qnormalize(
        torch.tensor([[-3.0, 2.0, 1.0, -0.5], [2.0, -1.0, 0.25, 0.75]], dtype=torch.float64)
    )

    assert torch.allclose(
        qnorm(qmul(a, b)),
        torch.ones(2, dtype=torch.float64),
        rtol=1e-12,
        atol=1e-12,
    )


def test_inverse_reverses_hamilton_product_order() -> None:
    a = torch.tensor(
        [[1.0, -2.0, 3.0, -4.0], [0.5, 1.5, -2.5, 3.5]], dtype=torch.float64
    )
    b = torch.tensor(
        [[-3.0, 2.0, 1.0, -0.5], [2.0, -1.0, 0.25, 0.75]], dtype=torch.float64
    )

    assert torch.allclose(
        qinverse(qmul(a, b)),
        qmul(qinverse(b), qinverse(a)),
        rtol=1e-12,
        atol=1e-12,
    )


def test_normalize_rejects_zero_quaternion() -> None:
    zero = torch.zeros(4)

    try:
        qnormalize(zero)
    except ValueError as exc:
        assert "too close to zero" in str(exc)
    else:
        raise AssertionError("qnormalize should reject zero quaternions")
