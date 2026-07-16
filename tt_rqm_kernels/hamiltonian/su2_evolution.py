"""Complete two-level Hamiltonian evolution reference path."""

from __future__ import annotations

import torch

from tt_rqm_kernels.hamiltonian.su2_compose import su2_compose_chain
from tt_rqm_kernels.hamiltonian.su2_lowering import lower_two_level_hamiltonian


def evolve_two_level_hamiltonian(
    hamiltonians: torch.Tensor,
    dt: float | torch.Tensor,
    *,
    hbar: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lower and compose a ``[B,K,4]`` two-level Hamiltonian chain.

    The result is a final rotor ``[B,4]`` in ``[w,x,y,z]`` order and phase
    pair ``[B,2]`` in ``[real,imag]`` order.  Composition is exactly
    ``step[K-1] * ... * step[0]`` and no normalization is applied.
    """

    rotors, phases = lower_two_level_hamiltonian(hamiltonians, dt, hbar=hbar)
    return su2_compose_chain(rotors, phases)
