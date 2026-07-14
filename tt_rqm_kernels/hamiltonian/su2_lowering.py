"""Closed-form lowering for piecewise-constant two-level Hamiltonians."""

from __future__ import annotations

import math

import torch


def _validate_hamiltonians(hamiltonians: torch.Tensor) -> None:
    if not isinstance(hamiltonians, torch.Tensor):
        raise TypeError("hamiltonians must be a torch.Tensor")
    if hamiltonians.ndim != 3 or hamiltonians.shape[-1] != 4:
        raise ValueError("hamiltonians must have shape [B, K, 4]")
    if hamiltonians.shape[0] < 1 or hamiltonians.shape[1] < 1:
        raise ValueError("hamiltonians require B >= 1 and K >= 1")
    if not hamiltonians.dtype.is_floating_point or hamiltonians.is_complex():
        raise TypeError("hamiltonians must use a real floating-point dtype")
    if not torch.isfinite(hamiltonians).all().item():
        raise ValueError("hamiltonians must contain only finite values")


def _broadcast_dt(
    dt: float | torch.Tensor,
    hamiltonians: torch.Tensor,
) -> torch.Tensor:
    if isinstance(dt, torch.Tensor) and dt.device != hamiltonians.device:
        raise ValueError("dt must be on the same device as hamiltonians")
    try:
        value = torch.as_tensor(dt, dtype=hamiltonians.dtype, device=hamiltonians.device)
        value = torch.broadcast_to(value, hamiltonians.shape[:2])
    except (TypeError, RuntimeError) as exc:
        raise ValueError("dt must be scalar or broadcastable to [B, K]") from exc
    if not torch.isfinite(value).all().item():
        raise ValueError("dt must contain only finite values")
    return value


def _validate_hbar(hbar: float) -> float:
    try:
        value = float(hbar)
    except (TypeError, ValueError) as exc:
        raise TypeError("hbar must be a finite positive scalar") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("hbar must be a finite positive scalar")
    return value


def lower_two_level_hamiltonian(
    hamiltonians: torch.Tensor,
    dt: float | torch.Tensor,
    *,
    hbar: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lower ``[h0, hx, hy, hz]`` coefficients to rotor and phase steps.

    The returned tensors have shapes ``[B, K, 4]`` and ``[B, K, 2]``.
    Phase pairs are complex ``[real, imag]`` values in
    ``[cos(alpha), -sin(alpha)]`` order. No normalization is applied.
    """

    _validate_hamiltonians(hamiltonians)
    step = _broadcast_dt(dt, hamiltonians) / _validate_hbar(hbar)

    h0 = hamiltonians[..., 0]
    vector = hamiltonians[..., 1:]
    magnitude = torch.linalg.vector_norm(vector, dim=-1)
    theta = magnitude * step
    safe_magnitude = torch.where(magnitude > 0.0, magnitude, torch.ones_like(magnitude))
    vector_scale = torch.sin(theta) / safe_magnitude
    vector_scale = torch.where(magnitude > 0.0, vector_scale, torch.zeros_like(vector_scale))

    rotor = torch.cat((torch.cos(theta).unsqueeze(-1), vector * vector_scale.unsqueeze(-1)), dim=-1)
    alpha = h0 * step
    phase = torch.stack((torch.cos(alpha), -torch.sin(alpha)), dim=-1)
    return rotor, phase
