"""Ordered CPU reference composition for pre-lowered SU(2) steps."""

from __future__ import annotations

import torch

from tt_rqm_kernels.quaternion_ops import qmul


def _validate_steps(rotors: torch.Tensor, phases: torch.Tensor) -> None:
    if not isinstance(rotors, torch.Tensor) or not isinstance(phases, torch.Tensor):
        raise TypeError("rotors and phases must be torch.Tensor values")
    if rotors.ndim != 3 or rotors.shape[-1] != 4:
        raise ValueError("rotors must have shape [B, K, 4]")
    if phases.ndim != 3 or phases.shape[-1] != 2:
        raise ValueError("phases must have shape [B, K, 2]")
    if rotors.shape[:2] != phases.shape[:2]:
        raise ValueError("rotors and phases must have matching B and K dimensions")
    if rotors.shape[0] < 1 or rotors.shape[1] < 1:
        raise ValueError("rotors and phases require B >= 1 and K >= 1")
    if rotors.dtype != phases.dtype or rotors.device != phases.device:
        raise ValueError("rotors and phases must share dtype and device")
    if not rotors.dtype.is_floating_point:
        raise TypeError("rotors and phases must use a floating-point dtype")
    if not torch.isfinite(rotors).all().item() or not torch.isfinite(phases).all().item():
        raise ValueError("rotors and phases must contain only finite values")


def _phase_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ar, ai = a.unbind(dim=-1)
    br, bi = b.unbind(dim=-1)
    return torch.stack((ar * br - ai * bi, ar * bi + ai * br), dim=-1)


def su2_compose_chain(
    rotors: torch.Tensor,
    phases: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose ``step[K-1] * ... * step[0]`` without renormalization."""

    _validate_steps(rotors, phases)
    total_rotor = rotors[:, 0].clone()
    total_phase = phases[:, 0].clone()
    for step in range(1, rotors.shape[1]):
        total_rotor = qmul(rotors[:, step], total_rotor)
        total_phase = _phase_mul(phases[:, step], total_phase)
    return total_rotor, total_phase
