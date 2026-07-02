"""Small independent scalar references for quaternion correctness checks."""

from __future__ import annotations

import math
from collections.abc import Sequence

Quaternion = tuple[float, float, float, float]
Vector3 = tuple[float, float, float]


def _quaternion(value: Sequence[float]) -> Quaternion:
    if len(value) != 4:
        raise ValueError(f"quaternion must have 4 values, got {len(value)}")
    return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))


def _vector3(value: Sequence[float]) -> Vector3:
    if len(value) != 3:
        raise ValueError(f"vector must have 3 values, got {len(value)}")
    return (float(value[0]), float(value[1]), float(value[2]))


def qmul_scalar(a: Sequence[float], b: Sequence[float]) -> Quaternion:
    """Hamilton product for two scalar quaternions."""

    ar, ai, aj, ak = _quaternion(a)
    br, bi, bj, bk = _quaternion(b)
    return (
        ar * br - ai * bi - aj * bj - ak * bk,
        ar * bi + ai * br + aj * bk - ak * bj,
        ar * bj - ai * bk + aj * br + ak * bi,
        ar * bk + ai * bj - aj * bi + ak * br,
    )


def qconj_scalar(q: Sequence[float]) -> Quaternion:
    """Quaternion conjugate."""

    real, i, j, k = _quaternion(q)
    return (real, -i, -j, -k)


def qnorm_scalar(q: Sequence[float]) -> float:
    """Euclidean quaternion norm."""

    real, i, j, k = _quaternion(q)
    return math.sqrt(real * real + i * i + j * j + k * k)


def qnormalize_scalar(q: Sequence[float], *, eps: float = 1e-12) -> Quaternion:
    """Normalize one scalar quaternion."""

    q_value = _quaternion(q)
    norm = qnorm_scalar(q_value)
    if norm <= eps:
        raise ValueError("q contains values too close to zero")
    return tuple(component / norm for component in q_value)


def qinverse_scalar(q: Sequence[float], *, eps: float = 1e-12) -> Quaternion:
    """Multiplicative inverse for one scalar quaternion."""

    q_value = _quaternion(q)
    norm_sq = sum(component * component for component in q_value)
    if norm_sq <= eps:
        raise ValueError("q contains values too close to zero")
    conj = qconj_scalar(q_value)
    return tuple(component / norm_sq for component in conj)


def qrotate_vector_scalar(rotor: Sequence[float], vector: Sequence[float]) -> Vector3:
    """Rotate one 3-vector by one unit quaternion rotor."""

    rotor_value = _quaternion(rotor)
    vector_value = _vector3(vector)
    pure_vector = (0.0, vector_value[0], vector_value[1], vector_value[2])
    rotated = qmul_scalar(qmul_scalar(rotor_value, pure_vector), qconj_scalar(rotor_value))
    return (rotated[1], rotated[2], rotated[3])
