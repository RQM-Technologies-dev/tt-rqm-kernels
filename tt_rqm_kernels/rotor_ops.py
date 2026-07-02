"""Rotor helpers built on unit quaternions."""

from __future__ import annotations

import torch

from tt_rqm_kernels.qtensor import validate_qtensor, validate_vector_tensor
from tt_rqm_kernels.quaternion_ops import qconj, qmul, qnorm


def vector_to_pure_quaternion(v: torch.Tensor) -> torch.Tensor:
    """Convert a 3-vector tensor `[x, y, z]` into `[0, x, y, z]`."""

    validate_vector_tensor(v, "v")
    real = torch.zeros_like(v[..., :1])
    return torch.cat((real, v), dim=-1)


def is_unit_rotor(
    rotor: torch.Tensor,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> bool:
    """Return whether every quaternion in `rotor` is unit length."""

    validate_qtensor(rotor, "rotor")
    norms = qnorm(rotor)
    return torch.allclose(norms, torch.ones_like(norms), atol=atol, rtol=rtol)


def validate_unit_rotor(
    rotor: torch.Tensor,
    *,
    name: str = "rotor",
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> torch.Tensor:
    """Validate that `rotor` is a unit quaternion tensor."""

    validate_qtensor(rotor, name)
    norms = qnorm(rotor)
    if not torch.allclose(norms, torch.ones_like(norms), atol=atol, rtol=rtol):
        max_error = torch.max(torch.abs(norms - 1.0)).item()
        raise ValueError(f"{name} must be unit length; max norm error {max_error:g}")
    return rotor


def qrotate_vector(
    rotor: torch.Tensor,
    vector: torch.Tensor,
    *,
    assume_unit: bool = False,
) -> torch.Tensor:
    """Rotate 3-vector tensors by unit quaternion rotors.

    `rotor` uses final dimension `4`; `vector` uses final dimension `3`.
    Leading dimensions broadcast through the underlying `qmul` calls.
    """

    if not assume_unit:
        validate_unit_rotor(rotor)
    else:
        validate_qtensor(rotor, "rotor")
    pure_vector = vector_to_pure_quaternion(vector)
    rotated = qmul(qmul(rotor, pure_vector), qconj(rotor))
    return rotated[..., 1:]
