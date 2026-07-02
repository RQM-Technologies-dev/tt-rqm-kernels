"""Validation and broadcasting helpers for structured tensor values."""

from __future__ import annotations

from collections.abc import Sequence

import torch

QUATERNION_DIM = 4
VECTOR_DIM = 3


def _require_tensor(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
    return value


def _require_floating(value: torch.Tensor, name: str) -> torch.Tensor:
    _require_tensor(value, name)
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must be a floating-point tensor")
    return value


def validate_qtensor(q: torch.Tensor, name: str = "q") -> torch.Tensor:
    """Validate a quaternion tensor with final dimension `[real, i, j, k]`."""

    _require_floating(q, name)
    if q.ndim < 1:
        raise ValueError(f"{name} must have at least one dimension")
    if q.shape[-1] != QUATERNION_DIM:
        raise ValueError(
            f"{name} must have final dimension size {QUATERNION_DIM}; "
            f"got shape {tuple(q.shape)}"
        )
    return q


def validate_vector_tensor(v: torch.Tensor, name: str = "v") -> torch.Tensor:
    """Validate a real vector tensor with final dimension `[x, y, z]`."""

    _require_floating(v, name)
    if v.ndim < 1:
        raise ValueError(f"{name} must have at least one dimension")
    if v.shape[-1] != VECTOR_DIM:
        raise ValueError(
            f"{name} must have final dimension size {VECTOR_DIM}; "
            f"got shape {tuple(v.shape)}"
        )
    return v


def leading_shape(value: torch.Tensor, name: str = "value") -> tuple[int, ...]:
    """Return all dimensions before the structured final dimension."""

    _require_tensor(value, name)
    if value.ndim < 1:
        raise ValueError(f"{name} must have at least one dimension")
    return tuple(value.shape[:-1])


def broadcast_leading_shapes(*shapes: Sequence[int]) -> tuple[int, ...]:
    """Broadcast leading shapes using PyTorch broadcasting rules."""

    normalized = [tuple(shape) for shape in shapes]
    if not normalized:
        return ()
    return tuple(torch.broadcast_shapes(*normalized))


def broadcast_qtensors(*qtensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Broadcast quaternion tensors over their leading dimensions."""

    validated = [validate_qtensor(q, f"qtensors[{idx}]") for idx, q in enumerate(qtensors)]
    if not validated:
        return ()

    shape = broadcast_leading_shapes(*(q.shape[:-1] for q in validated))
    return tuple(q.expand(*shape, QUATERNION_DIM) for q in validated)


def broadcast_vectors(*vectors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Broadcast 3-vector tensors over their leading dimensions."""

    validated = [validate_vector_tensor(v, f"vectors[{idx}]") for idx, v in enumerate(vectors)]
    if not validated:
        return ()

    shape = broadcast_leading_shapes(*(v.shape[:-1] for v in validated))
    return tuple(v.expand(*shape, VECTOR_DIM) for v in validated)


def broadcast_quaternion_and_vector(
    q: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Broadcast one quaternion tensor and one 3-vector tensor together."""

    validate_qtensor(q, "q")
    validate_vector_tensor(v, "v")
    shape = broadcast_leading_shapes(q.shape[:-1], v.shape[:-1])
    return q.expand(*shape, QUATERNION_DIM), v.expand(*shape, VECTOR_DIM)
