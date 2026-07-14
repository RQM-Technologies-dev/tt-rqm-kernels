"""Semantic diagnostics for pure two-qubit states."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tt_rqm_kernels.hamiltonian.two_qubit import _validate_state


@dataclass(frozen=True)
class TwoQubitStateDiagnostics:
    """Per-batch physical diagnostics without hidden state normalization."""

    state_norm_error: torch.Tensor
    density_trace_error: torch.Tensor
    hermiticity_error: torch.Tensor
    reduced_eigenvalues_a: torch.Tensor
    reduced_eigenvalues_b: torch.Tensor
    purity_a: torch.Tensor
    purity_b: torch.Tensor
    concurrence: torch.Tensor
    entropy_a: torch.Tensor
    entropy_b: torch.Tensor
    nonfinite_values: int


@dataclass(frozen=True)
class TwoQubitStateComparison:
    """Per-batch numerical error against a complex reference state."""

    max_abs_error: torch.Tensor
    rms_error: torch.Tensor
    global_phase_aligned_max_abs_error: torch.Tensor
    reduced_eigenvalue_max_abs_error: torch.Tensor


def _complex_state(state: torch.Tensor) -> torch.Tensor:
    return torch.complex(state[..., 0].double(), state[..., 1].double())


def _entropy(eigenvalues: torch.Tensor) -> torch.Tensor:
    probabilities = torch.clamp(eigenvalues.real, min=0.0)
    terms = torch.where(
        probabilities > 0.0,
        -probabilities * torch.log2(probabilities),
        torch.zeros_like(probabilities),
    )
    return terms.sum(dim=-1)


def two_qubit_state_diagnostics(state: torch.Tensor) -> TwoQubitStateDiagnostics:
    """Compute norm, reduced-state, concurrence, and entropy diagnostics."""

    if not isinstance(state, torch.Tensor):
        raise TypeError("state must be a torch.Tensor")
    if state.ndim != 3 or state.shape[-2:] != (4, 2):
        raise ValueError("state must have shape [B, 4, 2]")
    if not state.dtype.is_floating_point or state.is_complex():
        raise TypeError("state must use a real floating-point dtype")
    nonfinite_values = int((~torch.isfinite(state)).sum().item())
    if nonfinite_values:
        nan = torch.full((state.shape[0],), float("nan"), dtype=torch.float64, device=state.device)
        nan_eigenvalues = torch.full(
            (state.shape[0], 2),
            float("nan"),
            dtype=torch.float64,
            device=state.device,
        )
        return TwoQubitStateDiagnostics(
            state_norm_error=nan,
            density_trace_error=nan,
            hermiticity_error=nan,
            reduced_eigenvalues_a=nan_eigenvalues,
            reduced_eigenvalues_b=nan_eigenvalues.clone(),
            purity_a=nan.clone(),
            purity_b=nan.clone(),
            concurrence=nan.clone(),
            entropy_a=nan.clone(),
            entropy_b=nan.clone(),
            nonfinite_values=nonfinite_values,
        )
    _validate_state(state)

    psi = _complex_state(state)
    norm = torch.linalg.vector_norm(psi, dim=-1)
    density = torch.einsum("bi,bj->bij", psi, psi.conj())
    trace = torch.diagonal(density, dim1=-2, dim2=-1).sum(dim=-1).real
    hermiticity = torch.amax(torch.abs(density - density.transpose(-2, -1).conj()), dim=(-2, -1))

    amplitudes = psi.reshape(-1, 2, 2)
    reduced_a = amplitudes @ amplitudes.transpose(-2, -1).conj()
    reduced_b = torch.einsum("bik,bil->bkl", amplitudes, amplitudes.conj())
    eigenvalues_a = torch.linalg.eigvalsh(reduced_a).real
    eigenvalues_b = torch.linalg.eigvalsh(reduced_b).real
    purity_a = torch.diagonal(reduced_a @ reduced_a, dim1=-2, dim2=-1).sum(dim=-1).real
    purity_b = torch.diagonal(reduced_b @ reduced_b, dim1=-2, dim2=-1).sum(dim=-1).real
    a00, a01, a10, a11 = psi.unbind(dim=-1)
    concurrence = 2.0 * torch.abs(a00 * a11 - a01 * a10)

    return TwoQubitStateDiagnostics(
        state_norm_error=torch.abs(norm - 1.0),
        density_trace_error=torch.abs(trace - 1.0),
        hermiticity_error=hermiticity,
        reduced_eigenvalues_a=eigenvalues_a,
        reduced_eigenvalues_b=eigenvalues_b,
        purity_a=purity_a,
        purity_b=purity_b,
        concurrence=concurrence,
        entropy_a=_entropy(eigenvalues_a),
        entropy_b=_entropy(eigenvalues_b),
        nonfinite_values=0,
    )


def compare_two_qubit_states(
    actual: torch.Tensor,
    reference: torch.Tensor,
) -> TwoQubitStateComparison:
    """Compare real-lane output with a finite complex ``[B,4]`` reference."""

    _validate_state(actual)
    if not isinstance(reference, torch.Tensor) or reference.ndim != 2 or reference.shape[-1] != 4:
        raise ValueError("reference must be a complex tensor with shape [B, 4]")
    if not reference.is_complex():
        raise TypeError("reference must use a complex dtype")
    if reference.shape[0] != actual.shape[0] or reference.device != actual.device:
        raise ValueError("actual and reference must share batch and device")
    if not torch.isfinite(reference).all().item():
        raise ValueError("reference must contain only finite values")

    actual_complex = _complex_state(actual)
    reference_complex = reference.to(torch.complex128)
    difference = actual_complex - reference_complex
    overlap = torch.sum(reference_complex.conj() * actual_complex, dim=-1)
    magnitude = torch.abs(overlap)
    alignment = torch.where(
        magnitude > 0.0,
        overlap.conj() / magnitude,
        torch.ones_like(overlap),
    )
    aligned = actual_complex * alignment[:, None]

    reference_lanes = torch.stack((reference_complex.real, reference_complex.imag), dim=-1)
    actual_diagnostics = two_qubit_state_diagnostics(actual.double())
    reference_diagnostics = two_qubit_state_diagnostics(reference_lanes)
    eigenvalue_error = torch.amax(
        torch.cat(
            (
                torch.abs(
                    actual_diagnostics.reduced_eigenvalues_a
                    - reference_diagnostics.reduced_eigenvalues_a
                ),
                torch.abs(
                    actual_diagnostics.reduced_eigenvalues_b
                    - reference_diagnostics.reduced_eigenvalues_b
                ),
            ),
            dim=-1,
        ),
        dim=-1,
    )
    return TwoQubitStateComparison(
        max_abs_error=torch.amax(torch.abs(difference), dim=-1),
        rms_error=torch.sqrt(torch.mean(torch.abs(difference).square(), dim=-1)),
        global_phase_aligned_max_abs_error=torch.amax(
            torch.abs(aligned - reference_complex), dim=-1
        ),
        reduced_eigenvalue_max_abs_error=eigenvalue_error,
    )
