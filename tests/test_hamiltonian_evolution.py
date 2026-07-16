from __future__ import annotations

import math

import pytest
import torch

from tt_rqm_kernels import (
    compose_hamiltonian_matrices,
    evolve_two_level_hamiltonian,
    lower_two_level_hamiltonian,
    qmul,
    u2_matrix_from_rotor_phase,
)


def test_k1_and_zero_vector_global_phase_behavior() -> None:
    values = torch.tensor([[[2.0, 0.0, 0.0, 0.0]]], dtype=torch.float64)
    rotor, phase = evolve_two_level_hamiltonian(values, 0.25)
    assert torch.equal(rotor, torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64))
    assert torch.allclose(
        phase, torch.tensor([[math.cos(0.5), -math.sin(0.5)]], dtype=torch.float64)
    )


def test_scalar_and_exact_dt_are_equivalent() -> None:
    values = torch.randn((2, 8, 4), generator=torch.Generator().manual_seed(1))
    scalar = evolve_two_level_hamiltonian(values, 0.05)
    exact = evolve_two_level_hamiltonian(values, torch.full((2, 8), 0.05))
    assert torch.equal(scalar[0], exact[0])
    assert torch.equal(scalar[1], exact[1])


def test_noncommuting_order_and_reversal_sentinel() -> None:
    values = torch.tensor([[[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]], dtype=torch.float64)
    steps, _ = lower_two_level_hamiltonian(values, math.pi / 4)
    rotor, _ = evolve_two_level_hamiltonian(values, math.pi / 4)
    assert torch.allclose(rotor, qmul(steps[:, 1], steps[:, 0]))
    assert not torch.allclose(rotor, qmul(steps[:, 0], steps[:, 1]), atol=1e-6, rtol=1e-6)


def test_float64_agrees_with_independent_complex128_whole_chain_oracle() -> None:
    values = torch.randn(
        (3, 32, 4), generator=torch.Generator().manual_seed(2), dtype=torch.float64
    )
    rotor, phase = evolve_two_level_hamiltonian(values, 0.03)
    actual = u2_matrix_from_rotor_phase(rotor, phase)
    expected = compose_hamiltonian_matrices(values, 0.03)
    assert torch.allclose(actual, expected, atol=1e-11, rtol=1e-11)


def test_float32_chain_drift_is_not_normalized() -> None:
    values = torch.randn((4, 512, 4), generator=torch.Generator().manual_seed(3))
    rotor, phase = evolve_two_level_hamiltonian(values, 0.05)
    assert torch.max(torch.abs(torch.linalg.vector_norm(rotor, dim=-1) - 1.0)) > 0
    assert torch.max(torch.abs(torch.linalg.vector_norm(phase, dim=-1) - 1.0)) > 0


@pytest.mark.parametrize(
    ("values", "dt", "hbar"),
    [
        (torch.zeros((2, 4)), 0.1, 1.0),
        (torch.zeros((1, 0, 4)), 0.1, 1.0),
        (torch.zeros((1, 2, 4), dtype=torch.int64), 0.1, 1.0),
        (torch.full((1, 2, 4), math.nan), 0.1, 1.0),
        (torch.zeros((1, 2, 4)), torch.ones((3, 2)), 1.0),
        (torch.zeros((1, 2, 4)), 0.1, 0.0),
    ],
)
def test_invalid_inputs_match_h2a_rejection(
    values: torch.Tensor, dt: float | torch.Tensor, hbar: float
) -> None:
    with pytest.raises((TypeError, ValueError)):
        evolve_two_level_hamiltonian(values, dt, hbar=hbar)
