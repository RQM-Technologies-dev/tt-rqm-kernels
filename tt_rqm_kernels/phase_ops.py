"""Engineering-oriented phase and orientation tracking utilities."""

from __future__ import annotations

import math

import torch

from tt_rqm_kernels.quaternion_ops import qinverse, qmul, qnormalize
from tt_rqm_kernels.qtensor import validate_qtensor


def wrap_phase(phase: torch.Tensor, *, period: float = 2.0 * math.pi) -> torch.Tensor:
    """Wrap phase values into `[-period / 2, period / 2)`."""

    half_period = period / 2.0
    return torch.remainder(phase + half_period, period) - half_period


def phase_difference(
    current: torch.Tensor,
    previous: torch.Tensor,
    *,
    period: float = 2.0 * math.pi,
) -> torch.Tensor:
    """Wrapped difference `current - previous`."""

    return wrap_phase(current - previous, period=period)


def unwrap_phase(
    phases: torch.Tensor,
    *,
    dim: int = -1,
    period: float = 2.0 * math.pi,
) -> torch.Tensor:
    """Unwrap a phase sequence along one dimension."""

    if not isinstance(phases, torch.Tensor):
        raise TypeError(f"phases must be a torch.Tensor, got {type(phases).__name__}")
    moved = torch.movedim(phases, dim, -1)
    if moved.shape[-1] <= 1:
        return phases.clone()

    deltas = phase_difference(moved[..., 1:], moved[..., :-1], period=period)
    first = moved[..., :1]
    unwrapped = torch.cat((first, first + torch.cumsum(deltas, dim=-1)), dim=-1)
    return torch.movedim(unwrapped, -1, dim)


def integrate_phase(
    phase: torch.Tensor,
    angular_rate: torch.Tensor,
    dt: float | torch.Tensor,
    *,
    period: float = 2.0 * math.pi,
) -> torch.Tensor:
    """Advance a phase value by angular rate and timestep, then wrap it."""

    return wrap_phase(phase + angular_rate * dt, period=period)


def smooth_phase(
    previous: torch.Tensor,
    measurement: torch.Tensor,
    alpha: float | torch.Tensor,
    *,
    period: float = 2.0 * math.pi,
) -> torch.Tensor:
    """First-order smoother for wrapped phase measurements."""

    delta = phase_difference(measurement, previous, period=period)
    return wrap_phase(previous + alpha * delta, period=period)


def phase_to_unit_vector(phase: torch.Tensor) -> torch.Tensor:
    """Represent phase as `[cos(phase), sin(phase)]`."""

    return torch.stack((torch.cos(phase), torch.sin(phase)), dim=-1)


def rotor_orientation_error(current: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return the normalized rotor that maps `target` orientation to `current`."""

    validate_qtensor(current, "current")
    validate_qtensor(target, "target")
    error = qnormalize(qmul(qinverse(target), current))
    sign = torch.where(error[..., :1] < 0.0, -1.0, 1.0)
    return error * sign


def rotor_angle(rotor: torch.Tensor) -> torch.Tensor:
    """Return the shortest rotation angle represented by a rotor."""

    rotor = qnormalize(rotor)
    real = torch.abs(rotor[..., 0])
    vector_norm = torch.linalg.vector_norm(rotor[..., 1:], dim=-1)
    return 2.0 * torch.atan2(vector_norm, real)
