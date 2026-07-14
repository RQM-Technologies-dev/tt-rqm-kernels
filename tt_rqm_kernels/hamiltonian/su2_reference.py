"""Independent complex-matrix references for two-level evolution."""

from __future__ import annotations

import torch

from tt_rqm_kernels.hamiltonian.su2_lowering import (
    _broadcast_dt,
    _validate_hamiltonians,
    _validate_hbar,
)


def u2_matrix_from_rotor_phase(rotor: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """Convert ``[w,x,y,z]`` plus ``[real,imag]`` phase to a U(2) matrix."""

    if not isinstance(rotor, torch.Tensor) or not isinstance(phase, torch.Tensor):
        raise TypeError("rotor and phase must be torch.Tensor values")
    if rotor.shape[-1:] != (4,) or phase.shape[-1:] != (2,):
        raise ValueError("rotor and phase must end in dimensions 4 and 2")
    if rotor.shape[:-1] != phase.shape[:-1]:
        raise ValueError("rotor and phase leading dimensions must match")
    if rotor.dtype != phase.dtype or rotor.device != phase.device:
        raise ValueError("rotor and phase must share dtype and device")
    if not rotor.dtype.is_floating_point:
        raise TypeError("rotor and phase must use a floating-point dtype")

    w, x, y, z = rotor.unbind(dim=-1)
    top = torch.stack((torch.complex(w, -z), torch.complex(-y, -x)), dim=-1)
    bottom = torch.stack((torch.complex(y, -x), torch.complex(w, z)), dim=-1)
    su2 = torch.stack((top, bottom), dim=-2)
    scalar = torch.complex(phase[..., 0], phase[..., 1])
    return scalar[..., None, None] * su2


def compose_hamiltonian_matrices(
    hamiltonians: torch.Tensor,
    dt: float | torch.Tensor,
    *,
    hbar: float = 1.0,
) -> torch.Tensor:
    """Compose a complex128 matrix-exponential oracle in time order."""

    _validate_hamiltonians(hamiltonians)
    hbar_value = _validate_hbar(hbar)
    dt64 = _broadcast_dt(dt, hamiltonians).to(torch.float64)
    coefficients = hamiltonians.to(torch.float64)
    h0, hx, hy, hz = coefficients.unbind(dim=-1)
    matrix = torch.stack(
        (
            torch.stack((torch.complex(h0 + hz, torch.zeros_like(h0)), torch.complex(hx, -hy)), dim=-1),
            torch.stack((torch.complex(hx, hy), torch.complex(h0 - hz, torch.zeros_like(h0))), dim=-1),
        ),
        dim=-2,
    )
    steps = torch.linalg.matrix_exp((-1j * dt64[..., None, None] / hbar_value) * matrix)
    total = steps[:, 0]
    for step in range(1, steps.shape[1]):
        total = steps[:, step] @ total
    return total
