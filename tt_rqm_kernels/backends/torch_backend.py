"""Torch backend adapter for StructuredBench."""

from __future__ import annotations

import torch

from tt_rqm_kernels.phase_ops import integrate_phase, phase_to_unit_vector
from tt_rqm_kernels.quaternion_ops import (
    qconj,
    qdot,
    qinverse,
    qmul,
    qnorm,
    qnormalize,
)
from tt_rqm_kernels.rotor_ops import qrotate_vector

name = "torch"


def phase_update(
    phase: torch.Tensor,
    angular_rate: torch.Tensor,
    amplitude: torch.Tensor,
    dt: float | torch.Tensor,
) -> torch.Tensor:
    """Advance wrapped phase and return an amplitude-scaled `[cos, sin]` state."""

    next_phase = integrate_phase(phase, angular_rate, dt)
    return amplitude.unsqueeze(-1) * phase_to_unit_vector(next_phase)


__all__ = [
    "name",
    "phase_update",
    "qconj",
    "qdot",
    "qinverse",
    "qmul",
    "qnorm",
    "qnormalize",
    "qrotate_vector",
]
