"""Numerical conformance domain for the first H2B hardware pilot."""

from __future__ import annotations

import math

import torch

from tt_rqm_kernels.hamiltonian.su2_lowering import (
    _broadcast_dt,
    _validate_hamiltonians,
    _validate_hbar,
)

ROTOR_ANGLE_LIMIT = 1024.0
PHASE_ANGLE_LIMIT = 8192.0


class HamiltonianEvolutionDomainError(ValueError):
    """Raised when valid mathematics lies outside the frozen pilot domain."""


def angle_extrema(
    hamiltonians: torch.Tensor, dt: float | torch.Tensor, *, hbar: float = 1.0
) -> dict[str, float]:
    """Return maximum logical rotor and phase angle magnitudes in Float64."""

    _validate_hamiltonians(hamiltonians)
    hbar_value = _validate_hbar(hbar)
    coefficients = hamiltonians.detach().to(torch.float64)
    step = _broadcast_dt(dt, hamiltonians).detach().to(torch.float64) / hbar_value
    rotor = torch.linalg.vector_norm(coefficients[..., 1:], dim=-1) * torch.abs(step)
    phase = torch.abs(coefficients[..., 0] * step)
    return {
        "max_rotor_angle": float(rotor.max().item()),
        "max_phase_angle": float(phase.max().item()),
    }


def validate_pilot_domain(
    hamiltonians: torch.Tensor, dt: float | torch.Tensor, *, hbar: float = 1.0
) -> dict[str, float]:
    """Fail closed outside the H2B pilot domain without invalidating the input."""

    extrema = angle_extrema(hamiltonians, dt, hbar=hbar)
    if extrema["max_rotor_angle"] > ROTOR_ANGLE_LIMIT:
        raise HamiltonianEvolutionDomainError(
            "valid Hamiltonian is unsupported by the H2B pilot conformance contract: "
            f"rotor angle {extrema['max_rotor_angle']:.17g} exceeds {ROTOR_ANGLE_LIMIT:.17g}"
        )
    if extrema["max_phase_angle"] > PHASE_ANGLE_LIMIT:
        raise HamiltonianEvolutionDomainError(
            "valid Hamiltonian is unsupported by the H2B pilot conformance contract: "
            f"phase angle {extrema['max_phase_angle']:.17g} exceeds {PHASE_ANGLE_LIMIT:.17g}"
        )
    if not all(math.isfinite(value) for value in extrema.values()):
        raise HamiltonianEvolutionDomainError("H2B pilot angle extrema must be finite")
    return extrema
