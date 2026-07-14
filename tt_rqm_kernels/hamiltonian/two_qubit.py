"""CPU references for two-qubit Hamiltonian evolution in ordinary tensors."""

from __future__ import annotations

import torch

from tt_rqm_kernels.hamiltonian.su2_lowering import _broadcast_dt, _validate_hbar
from tt_rqm_kernels.hamiltonian.su2_reference import u2_matrix_from_rotor_phase


def _validate_two_qubit_hamiltonians(hamiltonians: torch.Tensor) -> None:
    if not isinstance(hamiltonians, torch.Tensor):
        raise TypeError("hamiltonians must be a torch.Tensor")
    if hamiltonians.ndim != 4 or hamiltonians.shape[-2:] != (4, 4):
        raise ValueError("hamiltonians must have shape [B, K, 4, 4]")
    if hamiltonians.shape[0] < 1 or hamiltonians.shape[1] < 1:
        raise ValueError("hamiltonians require B >= 1 and K >= 1")
    if not hamiltonians.dtype.is_floating_point or hamiltonians.is_complex():
        raise TypeError("hamiltonians must use a real floating-point dtype")
    if not torch.isfinite(hamiltonians).all().item():
        raise ValueError("hamiltonians must contain only finite values")


def _validate_state(state: torch.Tensor) -> None:
    if not isinstance(state, torch.Tensor):
        raise TypeError("state must be a torch.Tensor")
    if state.ndim != 3 or state.shape[-2:] != (4, 2):
        raise ValueError("state must have shape [B, 4, 2]")
    if state.shape[0] < 1:
        raise ValueError("state requires B >= 1")
    if not state.dtype.is_floating_point or state.is_complex():
        raise TypeError("state must use a real floating-point dtype")
    if not torch.isfinite(state).all().item():
        raise ValueError("state must contain only finite values")


def _pauli_basis(*, device: torch.device) -> torch.Tensor:
    zero = 0.0j
    one = 1.0 + 0.0j
    return torch.tensor(
        (
            ((one, zero), (zero, one)),
            ((zero, one), (one, zero)),
            ((zero, -1.0j), (1.0j, zero)),
            ((one, zero), (zero, -one)),
        ),
        dtype=torch.complex128,
        device=device,
    )


def two_qubit_hamiltonian_matrix(hamiltonians: torch.Tensor) -> torch.Tensor:
    """Expand real Pauli-product coefficients into complex128 matrices.

    ``hamiltonians[..., p, q]`` multiplies ``sigma[p] tensor sigma[q]`` with
    axis order ``[I, X, Y, Z]``. Every real coefficient tensor therefore
    represents a Hermitian two-qubit Hamiltonian.
    """

    _validate_two_qubit_hamiltonians(hamiltonians)
    pauli = _pauli_basis(device=hamiltonians.device)
    products = torch.einsum("aij,bkl->abikjl", pauli, pauli).reshape(4, 4, 4, 4)
    return torch.einsum("...ab,abij->...ij", hamiltonians.to(torch.complex128), products)


def lower_two_qubit_hamiltonian(
    hamiltonians: torch.Tensor,
    dt: float | torch.Tensor,
    *,
    hbar: float = 1.0,
) -> torch.Tensor:
    """CPU-lower ``[B,K,4,4]`` coefficients to ``[B,K,4,4,2]`` operators.

    The final dimension stores ``[real, imag]`` lanes. The result is cast back
    to the real input dtype so Float32 lowering models the serialized operator
    values a later device path would consume.
    """

    _validate_two_qubit_hamiltonians(hamiltonians)
    step = _broadcast_dt(dt, hamiltonians).to(torch.float64) / _validate_hbar(hbar)
    matrices = two_qubit_hamiltonian_matrix(hamiltonians)
    operators = torch.linalg.matrix_exp(-1j * step[..., None, None] * matrices)
    return torch.stack((operators.real, operators.imag), dim=-1).to(hamiltonians.dtype)


def compose_two_qubit_state(operators: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    """Apply ordered real-lane operators without normalizing the state.

    Applying step zero first and step ``K-1`` last produces
    ``U[K-1] ... U[0] |psi>``.
    """

    if not isinstance(operators, torch.Tensor):
        raise TypeError("operators must be a torch.Tensor")
    if operators.ndim != 5 or operators.shape[-3:] != (4, 4, 2):
        raise ValueError("operators must have shape [B, K, 4, 4, 2]")
    if operators.shape[0] < 1 or operators.shape[1] < 1:
        raise ValueError("operators require B >= 1 and K >= 1")
    if not operators.dtype.is_floating_point or operators.is_complex():
        raise TypeError("operators must use a real floating-point dtype")
    if not torch.isfinite(operators).all().item():
        raise ValueError("operators must contain only finite values")
    _validate_state(state)
    if operators.shape[0] != state.shape[0]:
        raise ValueError("operators and state must have matching B dimensions")
    if operators.dtype != state.dtype or operators.device != state.device:
        raise ValueError("operators and state must share dtype and device")

    real = state[..., 0].clone()
    imag = state[..., 1].clone()
    for step in range(operators.shape[1]):
        operator_real = operators[:, step, ..., 0]
        operator_imag = operators[:, step, ..., 1]
        next_real = torch.einsum("bij,bj->bi", operator_real, real) - torch.einsum(
            "bij,bj->bi", operator_imag, imag
        )
        next_imag = torch.einsum("bij,bj->bi", operator_real, imag) + torch.einsum(
            "bij,bj->bi", operator_imag, real
        )
        real, imag = next_real, next_imag
    return torch.stack((real, imag), dim=-1)


def evolve_two_qubit_state_reference(
    hamiltonians: torch.Tensor,
    state: torch.Tensor,
    dt: float | torch.Tensor,
    *,
    hbar: float = 1.0,
) -> torch.Tensor:
    """Independently evolve a state with complex128 matrix exponentials."""

    _validate_two_qubit_hamiltonians(hamiltonians)
    _validate_state(state)
    if hamiltonians.shape[0] != state.shape[0]:
        raise ValueError("hamiltonians and state must have matching B dimensions")
    if hamiltonians.device != state.device:
        raise ValueError("hamiltonians and state must share device")
    step = _broadcast_dt(dt, hamiltonians).to(torch.float64) / _validate_hbar(hbar)
    matrices = two_qubit_hamiltonian_matrix(hamiltonians)
    operators = torch.linalg.matrix_exp(-1j * step[..., None, None] * matrices)
    result = torch.complex(state[..., 0].double(), state[..., 1].double())
    for index in range(operators.shape[1]):
        result = torch.einsum("bij,bj->bi", operators[:, index], result)
    return result


def apply_local_rotor_pair(
    state: torch.Tensor,
    rotor_a: torch.Tensor,
    phase_a: torch.Tensor,
    rotor_b: torch.Tensor,
    phase_b: torch.Tensor,
) -> torch.Tensor:
    """Apply ``U_A tensor U_B`` using the existing rotor/phase convention."""

    _validate_state(state)
    values = (rotor_a, phase_a, rotor_b, phase_b)
    expected_shapes = ((state.shape[0], 4), (state.shape[0], 2)) * 2
    for value, expected in zip(values, expected_shapes, strict=True):
        if not isinstance(value, torch.Tensor):
            raise TypeError("rotors and phases must be torch.Tensor values")
        if value.shape != expected:
            raise ValueError("rotors and phases must match state batch and lane dimensions")
        if value.dtype != state.dtype or value.device != state.device:
            raise ValueError("state, rotors, and phases must share dtype and device")
        if not torch.isfinite(value).all().item():
            raise ValueError("rotors and phases must contain only finite values")

    unitary_a = u2_matrix_from_rotor_phase(rotor_a, phase_a)
    unitary_b = u2_matrix_from_rotor_phase(rotor_b, phase_b)
    local = torch.einsum("bij,bkl->bikjl", unitary_a, unitary_b).reshape(-1, 4, 4)
    complex_state = torch.complex(state[..., 0], state[..., 1])
    result = torch.einsum("bij,bj->bi", local, complex_state)
    return torch.stack((result.real, result.imag), dim=-1).to(state.dtype)
