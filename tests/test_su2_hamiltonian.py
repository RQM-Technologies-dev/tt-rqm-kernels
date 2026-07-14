from __future__ import annotations

import math

import pytest
import torch

from tt_rqm_kernels import (
    compose_hamiltonian_matrices,
    lower_two_level_hamiltonian,
    qmul,
    su2_compose_chain,
    u2_matrix_from_rotor_phase,
)


def test_zero_vector_hamiltonian_has_identity_rotor_and_global_phase() -> None:
    hamiltonians = torch.tensor([[[2.0, 0.0, 0.0, 0.0]]], dtype=torch.float64)
    rotor, phase = lower_two_level_hamiltonian(hamiltonians, 0.25)

    assert torch.equal(rotor, torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float64))
    assert torch.allclose(
        phase,
        torch.tensor([[[math.cos(0.5), -math.sin(0.5)]]], dtype=torch.float64),
        atol=1e-14,
        rtol=1e-14,
    )


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_axis_lowering_matches_expected_rotor(axis: int) -> None:
    coefficients = torch.zeros((1, 1, 4), dtype=torch.float64)
    coefficients[0, 0, axis + 1] = 2.0
    rotor, phase = lower_two_level_hamiltonian(coefficients, 0.125)
    expected = torch.zeros((1, 1, 4), dtype=torch.float64)
    expected[..., 0] = math.cos(0.25)
    expected[..., axis + 1] = math.sin(0.25)

    assert torch.allclose(rotor, expected, atol=1e-14, rtol=1e-14)
    assert torch.equal(phase, torch.tensor([[[1.0, -0.0]]], dtype=torch.float64))


def test_noncommuting_order_is_step_k_minus_one_through_step_zero() -> None:
    hamiltonians = torch.tensor(
        [[[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]], dtype=torch.float64
    )
    rotors, phases = lower_two_level_hamiltonian(hamiltonians, math.pi / 4.0)
    total_rotor, total_phase = su2_compose_chain(rotors, phases)

    expected = qmul(rotors[:, 1], rotors[:, 0])
    reversed_order = qmul(rotors[:, 0], rotors[:, 1])
    assert torch.allclose(total_rotor, expected, atol=1e-14, rtol=1e-14)
    assert not torch.allclose(total_rotor, reversed_order, atol=1e-6, rtol=1e-6)
    assert torch.equal(total_phase, torch.tensor([[1.0, -0.0]], dtype=torch.float64))


def test_complex128_oracle_agrees_with_float64_rotor_phase_composition() -> None:
    generator = torch.Generator().manual_seed(0)
    hamiltonians = torch.randn((7, 32, 4), generator=generator, dtype=torch.float64)
    rotors, phases = lower_two_level_hamiltonian(hamiltonians, 0.05)
    total_rotor, total_phase = su2_compose_chain(rotors, phases)
    quaternion_matrices = u2_matrix_from_rotor_phase(total_rotor, total_phase)
    exponential_matrices = compose_hamiltonian_matrices(hamiltonians, 0.05)

    assert quaternion_matrices.dtype == torch.complex128
    assert torch.allclose(quaternion_matrices, exponential_matrices, atol=1e-11, rtol=1e-11)


def test_float32_error_grows_without_hidden_renormalization() -> None:
    generator = torch.Generator().manual_seed(0)
    hamiltonians64 = torch.randn((8, 512, 4), generator=generator, dtype=torch.float64)
    rotors64, phases64 = lower_two_level_hamiltonian(hamiltonians64, 0.05)
    reference_rotor, reference_phase = su2_compose_chain(rotors64, phases64)
    result_rotor, result_phase = su2_compose_chain(rotors64.float(), phases64.float())

    assert torch.max(torch.abs(result_rotor.double() - reference_rotor)).item() < 1e-4
    assert torch.max(torch.abs(result_phase.double() - reference_phase)).item() < 1e-4
    assert torch.max(torch.abs(torch.linalg.vector_norm(result_rotor, dim=-1) - 1.0)).item() > 0.0
    assert torch.max(torch.abs(torch.linalg.vector_norm(result_phase, dim=-1) - 1.0)).item() > 0.0


def test_scalar_and_broadcast_dt_are_equivalent() -> None:
    hamiltonians = torch.ones((2, 3, 4), dtype=torch.float64)
    scalar = lower_two_level_hamiltonian(hamiltonians, 0.05)
    broadcast = lower_two_level_hamiltonian(
        hamiltonians, torch.full((2, 1), 0.05, dtype=torch.float64)
    )
    assert torch.equal(scalar[0], broadcast[0])
    assert torch.equal(scalar[1], broadcast[1])


@pytest.mark.parametrize(
    ("hamiltonians", "dt", "error"),
    [
        (torch.zeros((2, 4)), 0.1, ValueError),
        (torch.zeros((2, 3, 5)), 0.1, ValueError),
        (torch.zeros((2, 0, 4)), 0.1, ValueError),
        (torch.zeros((2, 3, 4), dtype=torch.int64), 0.1, TypeError),
        (torch.full((2, 3, 4), float("nan")), 0.1, ValueError),
        (torch.zeros((2, 3, 4)), torch.ones((4, 2)), ValueError),
    ],
)
def test_lowering_rejects_invalid_inputs(
    hamiltonians: torch.Tensor, dt: float | torch.Tensor, error: type[Exception]
) -> None:
    with pytest.raises(error):
        lower_two_level_hamiltonian(hamiltonians, dt)


@pytest.mark.parametrize("hbar", [0.0, -1.0, float("inf"), float("nan")])
def test_lowering_rejects_invalid_hbar(hbar: float) -> None:
    with pytest.raises(ValueError):
        lower_two_level_hamiltonian(torch.zeros((1, 1, 4)), 0.1, hbar=hbar)


def test_composition_rejects_mismatched_contracts() -> None:
    with pytest.raises(ValueError, match="matching B and K"):
        su2_compose_chain(torch.zeros((2, 3, 4)), torch.zeros((2, 4, 2)))
    with pytest.raises(ValueError, match="share dtype"):
        su2_compose_chain(
            torch.zeros((2, 3, 4), dtype=torch.float32),
            torch.zeros((2, 3, 2), dtype=torch.float64),
        )


def test_u2_global_phase_determinant_consistency() -> None:
    hamiltonians = torch.tensor([[[0.3, 0.4, -0.2, 0.7]]], dtype=torch.float64)
    rotors, phases = lower_two_level_hamiltonian(hamiltonians, 0.2)
    matrix = u2_matrix_from_rotor_phase(rotors[:, 0], phases[:, 0])
    phase = torch.complex(phases[:, 0, 0], phases[:, 0, 1])
    assert torch.allclose(torch.linalg.det(matrix), phase.square(), atol=1e-12, rtol=1e-12)
