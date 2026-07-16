"""Independent FP32 model for the device-side H2A compensated angle path."""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct


def fp32(value: float) -> float:
    """Round a Python float to IEEE binary32."""

    return struct.unpack("<f", struct.pack("<f", value))[0]


SPLITTER = fp32(4097.0)
TWO_PI_HI = fp32(2.0 * math.pi)
TWO_PI_LO = fp32(2.0 * math.pi - TWO_PI_HI)
INV_TWO_PI = fp32(1.0 / (2.0 * math.pi))


@dataclass(frozen=True)
class Float32Pair:
    hi: float
    lo: float

    @property
    def value(self) -> float:
        return self.hi + self.lo


def two_sum(lhs: float, rhs: float) -> Float32Pair:
    """Error-free TwoSum for binary32 operands, modeled with binary32 rounding."""

    lhs, rhs = fp32(lhs), fp32(rhs)
    hi = fp32(lhs + rhs)
    rhs_virtual = fp32(hi - lhs)
    lhs_virtual = fp32(hi - rhs_virtual)
    rhs_error = fp32(rhs - rhs_virtual)
    lhs_error = fp32(lhs - lhs_virtual)
    return Float32Pair(hi, fp32(lhs_error + rhs_error))


def split_two_product(lhs: float, rhs: float) -> Float32Pair:
    """Dekker TwoProduct used by the Wormhole kernel, without relying on FMA."""

    lhs, rhs = fp32(lhs), fp32(rhs)
    lhs_split = fp32(SPLITTER * lhs)
    lhs_hi = fp32(lhs_split - fp32(lhs_split - lhs))
    lhs_lo = fp32(lhs - lhs_hi)
    rhs_split = fp32(SPLITTER * rhs)
    rhs_hi = fp32(rhs_split - fp32(rhs_split - rhs))
    rhs_lo = fp32(rhs - rhs_hi)
    hi = fp32(lhs * rhs)
    lo = fp32(fp32(lhs_hi * rhs_hi) - hi)
    lo = fp32(lo + fp32(lhs_hi * rhs_lo))
    lo = fp32(lo + fp32(lhs_lo * rhs_hi))
    lo = fp32(lo + fp32(lhs_lo * rhs_lo))
    return Float32Pair(hi, lo)


def compensated_sum(values: tuple[float, ...]) -> Float32Pair:
    """Accumulate binary32 values as a high/low pair for Candidate C probes."""

    total = Float32Pair(0.0, 0.0)
    for value in values:
        primary = two_sum(total.hi, value)
        total = Float32Pair(primary.hi, fp32(total.lo + primary.lo))
    return total


def reduce_pair(angle: Float32Pair) -> float:
    """Reduce a high/low angle by split 2pi, collapsing only after reduction."""

    quotient = fp32(round(fp32(angle.hi * INV_TWO_PI)))
    period = split_two_product(quotient, fp32(-TWO_PI_HI))
    reduced_hi = fp32(angle.hi + period.hi)
    reduced_lo = fp32(angle.lo + period.lo)
    reduced_lo = fp32(reduced_lo + fp32(quotient * fp32(-TWO_PI_LO)))
    return fp32(reduced_hi + reduced_lo)


def compensated_angle(coefficient: float, scaled_dt: float) -> tuple[Float32Pair, float]:
    """Form and reduce an FP32-input angle without a host-precomputed angle."""

    pair = split_two_product(coefficient, scaled_dt)
    return pair, reduce_pair(pair)
