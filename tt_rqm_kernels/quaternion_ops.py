"""CPU/PyTorch reference quaternion operations."""

from __future__ import annotations

import torch

from tt_rqm_kernels.qtensor import broadcast_qtensors, validate_qtensor


def _raise_if_near_zero(value: torch.Tensor, eps: float, name: str) -> None:
    if torch.any(value <= eps).item():
        raise ValueError(f"{name} contains values too close to zero")


def qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two quaternion tensors.

    Inputs must have final dimension `4` in `[real, i, j, k]` order. Leading
    dimensions are broadcast with normal PyTorch broadcasting rules.
    """

    a, b = broadcast_qtensors(a, b)
    ar, ai, aj, ak = a.unbind(dim=-1)
    br, bi, bj, bk = b.unbind(dim=-1)

    real = ar * br - ai * bi - aj * bj - ak * bk
    i = ar * bi + ai * br + aj * bk - ak * bj
    j = ar * bj - ai * bk + aj * br + ak * bi
    k = ar * bk + ai * bj - aj * bi + ak * br
    return torch.stack((real, i, j, k), dim=-1)


def qconj(q: torch.Tensor) -> torch.Tensor:
    """Quaternion conjugate."""

    validate_qtensor(q, "q")
    signs = q.new_tensor((1.0, -1.0, -1.0, -1.0))
    return q * signs


def qdot(a: torch.Tensor, b: torch.Tensor, *, keepdim: bool = False) -> torch.Tensor:
    """Euclidean dot product over quaternion components."""

    a, b = broadcast_qtensors(a, b)
    return torch.sum(a * b, dim=-1, keepdim=keepdim)


def qnorm(q: torch.Tensor, *, keepdim: bool = False) -> torch.Tensor:
    """Euclidean norm of a quaternion tensor."""

    validate_qtensor(q, "q")
    return torch.linalg.vector_norm(q, dim=-1, keepdim=keepdim)


def qnormalize(q: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """Normalize a quaternion tensor.

    Raises `ValueError` when any quaternion has norm less than or equal to
    `eps`, because a zero quaternion has no unit orientation.
    """

    norms = qnorm(q, keepdim=True)
    _raise_if_near_zero(norms, eps, "q")
    return q / norms


def qinverse(q: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """Multiplicative inverse of a nonzero quaternion tensor."""

    validate_qtensor(q, "q")
    norm_sq = qdot(q, q, keepdim=True)
    _raise_if_near_zero(norm_sq, eps, "q")
    return qconj(q) / norm_sq
