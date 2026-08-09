"""Matched vectorized CPU formulations for QSP structured operators."""

from __future__ import annotations

from typing import Literal

import torch

from tt_rqm_kernels.quaternion_ops import qconj, qmul, qnormalize

Formulation = Literal[
    "unrestricted",
    "materialized_structured",
    "direct_structured_real",
    "complex_pair",
    "quaternion",
]


def quaternion_left_matrix(q: torch.Tensor) -> torch.Tensor:
    """Materialize the 4x4 real matrix for left Hamilton multiplication."""

    a, b, c, d = q.unbind(dim=-1)
    return torch.stack(
        (
            torch.stack((a, -b, -c, -d), dim=-1),
            torch.stack((b, a, -d, c), dim=-1),
            torch.stack((c, d, a, -b), dim=-1),
            torch.stack((d, -c, b, a), dim=-1),
        ),
        dim=-2,
    )


def j_involution(q: torch.Tensor) -> torch.Tensor:
    """Return q^j=[a,-b,c,-d] under H+Vj packing."""

    return q * q.new_tensor((1.0, -1.0, 1.0, -1.0))


def su2_from_quaternion(rotor: torch.Tensor) -> torch.Tensor:
    """Map normalized [w,x,y,z] rotors to canonical complex SU(2)."""

    return _complex_matrix(qnormalize(rotor))


def _complex_matrix(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(dim=-1)
    top = torch.stack((torch.complex(w, -z), torch.complex(-y, -x)), dim=-1)
    bottom = torch.stack((torch.complex(y, -x), torch.complex(w, z)), dim=-1)
    return torch.stack((top, bottom), dim=-2)


def run_operator(
    operator: str,
    formulation: Formulation,
    tensors: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """Run one formulation with identical tensors, dtype, and output semantics."""

    if operator == "qmul":
        a, b = tensors
        return _multiply(a, b, formulation)
    if operator == "qdot":
        x, weights = tensors
        return torch.sum(_multiply(x, weights, formulation), dim=-2)
    if operator == "su2-frame-transport":
        rotor, spinor = tensors
        unitary = su2_from_quaternion(rotor)
        if formulation in {"unrestricted", "materialized_structured"}:
            real_input = torch.view_as_real(spinor).reshape(spinor.shape[0], 4)
            real_matrix = torch.empty(
                (unitary.shape[0], 4, 4), dtype=unitary.real.dtype, device=unitary.device
            )
            for row in range(2):
                for column in range(2):
                    value = unitary[:, row, column]
                    real_matrix[:, 2 * row, 2 * column] = value.real
                    real_matrix[:, 2 * row, 2 * column + 1] = -value.imag
                    real_matrix[:, 2 * row + 1, 2 * column] = value.imag
                    real_matrix[:, 2 * row + 1, 2 * column + 1] = value.real
            return torch.einsum("nij,nj->ni", real_matrix, real_input).reshape(-1, 2, 2)
        if formulation == "direct_structured_real":
            w, x, y, z = qnormalize(rotor).unbind(dim=-1)
            first, second = spinor.unbind(dim=-1)
            out0 = torch.complex(w, -z) * first + torch.complex(-y, -x) * second
            out1 = torch.complex(y, -x) * first + torch.complex(w, z) * second
            return torch.view_as_real(torch.stack((out0, out1), dim=-1))
        return torch.view_as_real(unitary @ spinor.unsqueeze(-1)).squeeze(-2)
    if operator == "selective-q-plus-qj":
        x, w0, wj = tensors
        first = _multiply(x, w0, formulation)
        second = _multiply(j_involution(x), wj, formulation)
        return torch.sum(first + second, dim=-2)
    if operator == "fused-frame-filter":
        rotor, x, w0, wj = tensors
        filtered = torch.sum(
            _multiply(x, w0, formulation) + _multiply(j_involution(x), wj, formulation),
            dim=-2,
        )
        return _multiply(_multiply(rotor, filtered, formulation), qconj(rotor), formulation)
    if operator == "ordered-rotor-composition":
        (rotors,) = tensors
        result = rotors[:, 0]
        for index in range(1, rotors.shape[1]):
            result = _multiply(rotors[:, index], result, formulation)
        return result
    raise ValueError(f"unknown QSP operator: {operator}")


def make_operator_inputs(
    operator: str,
    size: int,
    *,
    seed: int,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, ...]:
    """Create identical deterministic inputs for every comparator."""

    generator = torch.Generator().manual_seed(seed)
    if operator == "qmul":
        return (
            torch.randn(size, 4, generator=generator, dtype=dtype),
            torch.randn(size, 4, generator=generator, dtype=dtype),
        )
    if operator in {"qdot", "selective-q-plus-qj"}:
        x = torch.randn(size, 4, 4, generator=generator, dtype=dtype)
        w0 = torch.randn(size, 4, 4, generator=generator, dtype=dtype)
        if operator == "qdot":
            return x, w0
        wj = torch.randn(size, 4, 4, generator=generator, dtype=dtype)
        return x, w0, wj
    if operator == "su2-frame-transport":
        rotor = qnormalize(torch.randn(size, 4, generator=generator, dtype=dtype))
        real = torch.randn(size, 2, generator=generator, dtype=dtype)
        imag = torch.randn(size, 2, generator=generator, dtype=dtype)
        return rotor, torch.complex(real, imag)
    if operator == "fused-frame-filter":
        rotor = qnormalize(torch.randn(size, 4, generator=generator, dtype=dtype))
        x = torch.randn(size, 4, 4, generator=generator, dtype=dtype)
        w0 = torch.randn(size, 4, 4, generator=generator, dtype=dtype)
        wj = torch.randn(size, 4, 4, generator=generator, dtype=dtype)
        return rotor, x, w0, wj
    if operator == "ordered-rotor-composition":
        rotors = qnormalize(torch.randn(size, 8, 4, generator=generator, dtype=dtype))
        return (rotors,)
    raise ValueError(f"unknown QSP operator: {operator}")


def coefficient_counts(operator: str) -> dict[str, int]:
    """Frozen logical coefficient counts for registered formulations."""

    if operator in {"qmul", "su2-frame-transport", "ordered-rotor-composition"}:
        unrestricted, structured = 16, 4
    elif operator == "qdot":
        unrestricted, structured = 64, 16
    else:
        unrestricted, structured = 128, 32
    return {
        "complex_pair": structured,
        "direct_structured_real": structured,
        "materialized_structured": unrestricted,
        "quaternion": structured,
        "unrestricted": unrestricted,
    }


def _multiply(a: torch.Tensor, b: torch.Tensor, formulation: Formulation) -> torch.Tensor:
    if formulation in {"unrestricted", "materialized_structured"}:
        return torch.einsum("...ij,...j->...i", quaternion_left_matrix(a), b)
    if formulation == "complex_pair":
        product = _complex_matrix(a) @ _complex_matrix(b)
        entry00 = product[..., 0, 0]
        entry01 = product[..., 0, 1]
        return torch.stack((entry00.real, -entry01.imag, -entry01.real, -entry00.imag), dim=-1)
    # Both paths are direct component expansions in the matched CPU harness.
    return qmul(a, b)
