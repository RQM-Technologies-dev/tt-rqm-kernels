from __future__ import annotations

import math

import pytest
import torch

from tt_rqm_kernels import (
    apply_local_rotor_pair,
    compare_two_qubit_states,
    compose_two_qubit_state,
    evolve_two_qubit_state_reference,
    lower_two_level_hamiltonian,
    lower_two_qubit_hamiltonian,
    two_qubit_hamiltonian_matrix,
    two_qubit_state_diagnostics,
)


def _state(amplitudes: list[complex], *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    value = torch.tensor(amplitudes, dtype=torch.complex128)
    return torch.stack((value.real, value.imag), dim=-1).to(dtype).unsqueeze(0)


def _basis(index: int, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    amplitudes = [0.0j] * 4
    amplitudes[index] = 1.0 + 0.0j
    return _state(amplitudes, dtype=dtype)


def _coefficients(*, dtype: torch.dtype = torch.float64, steps: int = 1) -> torch.Tensor:
    return torch.zeros((1, steps, 4, 4), dtype=dtype)


def test_pauli_product_coefficients_expand_to_expected_matrices() -> None:
    coefficients = _coefficients()
    coefficients[0, 0, 1, 1] = 1.0
    matrix = two_qubit_hamiltonian_matrix(coefficients)[0, 0]
    x = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex128)

    assert matrix.dtype == torch.complex128
    assert torch.equal(matrix, torch.kron(x, x))


def test_identity_coefficient_is_only_a_global_phase() -> None:
    coefficients = _coefficients()
    coefficients[..., 0, 0] = 2.0
    initial = _basis(0)
    result = evolve_two_qubit_state_reference(coefficients, initial, 0.25)

    expected_phase = torch.exp(torch.tensor(-0.5j, dtype=torch.complex128))
    assert torch.allclose(result[0, 0], expected_phase, atol=1e-13, rtol=1e-13)
    assert torch.equal(result[0, 1:], torch.zeros(3, dtype=torch.complex128))


def test_real_lane_composition_matches_independent_complex128_oracle() -> None:
    generator = torch.Generator().manual_seed(0)
    coefficients = torch.randn((5, 12, 4, 4), generator=generator, dtype=torch.float64)
    initial_complex = torch.randn(
        (5, 4), generator=generator, dtype=torch.float64
    ) + 1j * torch.randn((5, 4), generator=generator, dtype=torch.float64)
    initial_complex = initial_complex / torch.linalg.vector_norm(
        initial_complex, dim=-1, keepdim=True
    )
    initial = torch.stack((initial_complex.real, initial_complex.imag), dim=-1)

    operators = lower_two_qubit_hamiltonian(coefficients, 0.05)
    actual = compose_two_qubit_state(operators, initial)
    reference = evolve_two_qubit_state_reference(coefficients, initial, 0.05)
    comparison = compare_two_qubit_states(actual, reference)

    assert torch.max(comparison.max_abs_error).item() < 1e-11
    assert torch.max(comparison.reduced_eigenvalue_max_abs_error).item() < 1e-11


def test_noncommuting_step_order_is_observable() -> None:
    coefficients = _coefficients(steps=2)
    coefficients[0, 0, 1, 0] = 1.0
    coefficients[0, 1, 3, 3] = 1.0
    initial = _state([0.5, 0.5, 0.5, 0.5])
    forward = compose_two_qubit_state(lower_two_qubit_hamiltonian(coefficients, 0.4), initial)
    reversed_steps = compose_two_qubit_state(
        lower_two_qubit_hamiltonian(coefficients.flip(1), 0.4), initial
    )

    assert not torch.allclose(forward, reversed_steps, atol=1e-8, rtol=1e-8)


def test_product_and_bell_states_have_expected_entanglement_metrics() -> None:
    root_half = math.sqrt(0.5)
    states = torch.cat(
        (
            _basis(0),
            _basis(1),
            _state([root_half, 0.0, root_half, 0.0]),
            _state([root_half, 0.0, 0.0, root_half]),
            _state([root_half, 0.0, 0.0, -root_half]),
            _state([0.0, root_half, root_half, 0.0]),
            _state([0.0, root_half, -root_half, 0.0]),
        ),
        dim=0,
    )
    metrics = two_qubit_state_diagnostics(states)

    assert torch.allclose(metrics.concurrence[:3], torch.zeros(3, dtype=torch.float64))
    assert torch.allclose(metrics.entropy_a[:3], torch.zeros(3, dtype=torch.float64))
    assert torch.allclose(metrics.concurrence[3:], torch.ones(4, dtype=torch.float64))
    assert torch.allclose(metrics.entropy_a[3:], torch.ones(4, dtype=torch.float64))
    assert torch.allclose(metrics.purity_a[3:], torch.full((4,), 0.5, dtype=torch.float64))


def test_local_rotor_pairs_preserve_concurrence_and_reduced_spectra() -> None:
    root_half = math.sqrt(0.5)
    bell = _state([root_half, 0.0, 0.0, root_half])
    local_a = torch.tensor([[[0.2, 0.7, -0.1, 0.4]]], dtype=torch.float64)
    local_b = torch.tensor([[[-0.3, 0.1, 0.5, -0.2]]], dtype=torch.float64)
    rotor_a, phase_a = lower_two_level_hamiltonian(local_a, 0.3)
    rotor_b, phase_b = lower_two_level_hamiltonian(local_b, 0.3)
    transformed = apply_local_rotor_pair(
        bell,
        rotor_a[:, 0],
        phase_a[:, 0],
        rotor_b[:, 0],
        phase_b[:, 0],
    )
    before = two_qubit_state_diagnostics(bell)
    after = two_qubit_state_diagnostics(transformed)

    assert torch.allclose(after.concurrence, before.concurrence, atol=1e-12, rtol=1e-12)
    assert torch.allclose(
        after.reduced_eigenvalues_a,
        before.reduced_eigenvalues_a,
        atol=1e-12,
        rtol=1e-12,
    )
    assert torch.allclose(
        after.reduced_eigenvalues_b,
        before.reduced_eigenvalues_b,
        atol=1e-12,
        rtol=1e-12,
    )


@pytest.mark.parametrize(
    ("terms", "initial"),
    [
        (((1, 1),), _basis(0)),
        (((2, 2),), _basis(0)),
        (((3, 3),), _state([0.5, 0.5, 0.5, 0.5])),
        (((1, 1), (2, 2), (3, 3)), _basis(1)),
    ],
)
def test_nonlocal_interactions_generate_entanglement(
    terms: tuple[tuple[int, int], ...], initial: torch.Tensor
) -> None:
    coefficients = _coefficients()
    for first, second in terms:
        coefficients[0, 0, first, second] = 1.0
    dt = math.pi / 8.0 if len(terms) == 3 else math.pi / 4.0
    result = compose_two_qubit_state(lower_two_qubit_hamiltonian(coefficients, dt), initial)
    metrics = two_qubit_state_diagnostics(result)

    assert torch.allclose(metrics.concurrence, torch.ones(1, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(metrics.state_norm_error, torch.zeros(1, dtype=torch.float64), atol=1e-12)


def test_float32_chain_records_drift_without_hidden_normalization() -> None:
    generator = torch.Generator().manual_seed(0)
    coefficients64 = torch.randn((3, 128, 4, 4), generator=generator, dtype=torch.float64) * 0.2
    initial64 = _basis(0).expand(3, -1, -1).clone()
    reference = evolve_two_qubit_state_reference(coefficients64, initial64, 0.05)
    actual = compose_two_qubit_state(
        lower_two_qubit_hamiltonian(coefficients64.float(), 0.05), initial64.float()
    )
    comparison = compare_two_qubit_states(actual, reference)
    diagnostics = two_qubit_state_diagnostics(actual)

    assert torch.max(comparison.max_abs_error).item() < 1e-4
    assert torch.max(diagnostics.state_norm_error).item() > 0.0
    assert diagnostics.nonfinite_values == 0


def test_primary_composition_never_normalizes_input() -> None:
    identity = _coefficients()
    unnormalized = 2.0 * _basis(0)
    result = compose_two_qubit_state(lower_two_qubit_hamiltonian(identity, 0.1), unnormalized)
    metrics = two_qubit_state_diagnostics(result)

    assert torch.equal(result, unnormalized)
    assert torch.equal(metrics.state_norm_error, torch.ones(1, dtype=torch.float64))


def test_global_phase_aligned_error_distinguishes_physically_equal_states() -> None:
    reference = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.complex128)
    phase = torch.exp(torch.tensor(0.7j, dtype=torch.complex128))
    actual_complex = phase * reference
    actual = torch.stack((actual_complex.real, actual_complex.imag), dim=-1)
    comparison = compare_two_qubit_states(actual, reference)

    assert comparison.max_abs_error.item() > 0.1
    assert comparison.global_phase_aligned_max_abs_error.item() < 1e-12


def test_diagnostics_report_nonfinite_values_without_running_eigendecomposition() -> None:
    state = _basis(0)
    state[0, 1, 0] = float("nan")
    metrics = two_qubit_state_diagnostics(state)

    assert metrics.nonfinite_values == 1
    assert torch.isnan(metrics.concurrence).all()


@pytest.mark.parametrize(
    ("coefficients", "state", "dt", "error"),
    [
        (torch.zeros((1, 4, 4)), _basis(0), 0.1, ValueError),
        (torch.zeros((1, 0, 4, 4)), _basis(0), 0.1, ValueError),
        (torch.zeros((1, 1, 4, 5)), _basis(0), 0.1, ValueError),
        (torch.zeros((1, 1, 4, 4), dtype=torch.int64), _basis(0), 0.1, TypeError),
        (torch.full((1, 1, 4, 4), float("nan")), _basis(0), 0.1, ValueError),
        (_coefficients(), torch.zeros((1, 8)), 0.1, ValueError),
        (_coefficients(), _basis(0), torch.ones((2, 3)), ValueError),
    ],
)
def test_two_qubit_evolution_rejects_malformed_inputs(
    coefficients: torch.Tensor,
    state: torch.Tensor,
    dt: float | torch.Tensor,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        evolve_two_qubit_state_reference(coefficients, state, dt)


@pytest.mark.parametrize("hbar", [0.0, -1.0, float("inf"), float("nan")])
def test_two_qubit_lowering_rejects_invalid_hbar(hbar: float) -> None:
    with pytest.raises(ValueError):
        lower_two_qubit_hamiltonian(_coefficients(), 0.1, hbar=hbar)
