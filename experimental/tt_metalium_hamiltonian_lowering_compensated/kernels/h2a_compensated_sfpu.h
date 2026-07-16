// SPDX-FileCopyrightText: 2026 RQM Technologies LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

#include "sfpi.h"

namespace ckernel::sfpu {

constexpr std::size_t kVectorsPerFace = 8;
constexpr std::size_t kVectorsPerTile = 32;

inline sfpi::vFloat fused_multiply_add(sfpi::vFloat a, sfpi::vFloat b, sfpi::vFloat c) {
    return __builtin_rvtt_sfpmad(a.get(), b.get(), c.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
}

inline void split_product(sfpi::vFloat a, sfpi::vFloat b, sfpi::vFloat& hi, sfpi::vFloat& lo) {
    constexpr float splitter = 4097.0f;
    sfpi::vFloat a_split = splitter * a;
    sfpi::vFloat a_hi = a_split - (a_split - a);
    sfpi::vFloat a_lo = a - a_hi;
    sfpi::vFloat b_split = splitter * b;
    sfpi::vFloat b_hi = b_split - (b_split - b);
    sfpi::vFloat b_lo = b - b_hi;
    hi = a * b;
    lo = a_hi * b_hi - hi;
    lo = lo + a_hi * b_lo;
    lo = lo + a_lo * b_hi;
    lo = lo + a_lo * b_lo;
}

inline void product_face(uint32_t lhs, uint32_t rhs, uint32_t out) {
    const uint32_t lhs_base = lhs * kVectorsPerTile;
    const uint32_t rhs_base = rhs * kVectorsPerTile;
    const uint32_t out_base = out * kVectorsPerTile;
    for (std::size_t i = 0; i < kVectorsPerFace; ++i) {
        sfpi::vFloat a = sfpi::dst_reg[lhs_base + i];
        sfpi::vFloat b = sfpi::dst_reg[rhs_base + i];
        sfpi::dst_reg[out_base + i] = a * b;
    }
}

inline void add_face(uint32_t lhs, uint32_t rhs, uint32_t out) {
    const uint32_t lhs_base = lhs * kVectorsPerTile;
    const uint32_t rhs_base = rhs * kVectorsPerTile;
    const uint32_t out_base = out * kVectorsPerTile;
    for (std::size_t i = 0; i < kVectorsPerFace; ++i) {
        sfpi::vFloat a = sfpi::dst_reg[lhs_base + i];
        sfpi::vFloat b = sfpi::dst_reg[rhs_base + i];
        sfpi::dst_reg[out_base + i] = a + b;
    }
}

inline void two_product_face(uint32_t lhs, uint32_t rhs, uint32_t out) {
    const uint32_t lhs_base = lhs * kVectorsPerTile;
    const uint32_t rhs_base = rhs * kVectorsPerTile;
    const uint32_t hi_base = out * kVectorsPerTile;
    const uint32_t lo_base = (out + 1) * kVectorsPerTile;
    for (std::size_t i = 0; i < kVectorsPerFace; ++i) {
        sfpi::vFloat a = sfpi::dst_reg[lhs_base + i];
        sfpi::vFloat b = sfpi::dst_reg[rhs_base + i];
        sfpi::vFloat hi = a * b;
        sfpi::vFloat lo = fused_multiply_add(a, b, -hi);
        sfpi::dst_reg[hi_base + i] = hi;
        sfpi::dst_reg[lo_base + i] = lo;
    }
}

inline void split_two_product_face(uint32_t lhs, uint32_t rhs, uint32_t out) {
    const uint32_t lhs_base = lhs * kVectorsPerTile;
    const uint32_t rhs_base = rhs * kVectorsPerTile;
    const uint32_t hi_base = out * kVectorsPerTile;
    const uint32_t lo_base = (out + 1) * kVectorsPerTile;
    for (std::size_t i = 0; i < kVectorsPerFace; ++i) {
        sfpi::vFloat a = sfpi::dst_reg[lhs_base + i];
        sfpi::vFloat b = sfpi::dst_reg[rhs_base + i];
        sfpi::vFloat hi;
        sfpi::vFloat lo;
        split_product(a, b, hi, lo);
        sfpi::dst_reg[hi_base + i] = hi;
        sfpi::dst_reg[lo_base + i] = lo;
    }
}

inline void compensated_product_face(uint32_t coefficient, uint32_t step_hi, uint32_t step_lo, uint32_t out) {
    const uint32_t coefficient_base = coefficient * kVectorsPerTile;
    const uint32_t step_hi_base = step_hi * kVectorsPerTile;
    const uint32_t step_lo_base = step_lo * kVectorsPerTile;
    const uint32_t hi_base = out * kVectorsPerTile;
    const uint32_t lo_base = (out + 1) * kVectorsPerTile;
    for (std::size_t i = 0; i < kVectorsPerFace; ++i) {
        sfpi::vFloat a = sfpi::dst_reg[coefficient_base + i];
        sfpi::vFloat b_hi = sfpi::dst_reg[step_hi_base + i];
        sfpi::vFloat b_lo = sfpi::dst_reg[step_lo_base + i];
        sfpi::vFloat angle_hi;
        sfpi::vFloat angle_lo;
        split_product(a, b_hi, angle_hi, angle_lo);
        angle_lo = angle_lo + a * b_lo;
        sfpi::dst_reg[hi_base + i] = angle_hi;
        sfpi::dst_reg[lo_base + i] = angle_lo;
    }
}

inline void compensated_reduce_face(uint32_t angle_hi_tile, uint32_t angle_lo_tile, uint32_t out) {
    constexpr float inverse_two_pi = 0x1.45f306p-3f;
    constexpr float negative_two_pi_hi = -0x1.921fb6p+2f;
    constexpr float negative_two_pi_lo = 0x1.777a5cp-23f;
    constexpr float negative_two_pi_head = -0x1.922p+2f;
    constexpr float negative_two_pi_tail = 0x1.28p-16f;
    constexpr float rounding_bias_value = 0x1.8p23f;
    const uint32_t angle_hi_base = angle_hi_tile * kVectorsPerTile;
    const uint32_t angle_lo_base = angle_lo_tile * kVectorsPerTile;
    const uint32_t out_base = out * kVectorsPerTile;
    for (std::size_t i = 0; i < kVectorsPerFace; ++i) {
        sfpi::vFloat angle_hi = sfpi::dst_reg[angle_hi_base + i];
        sfpi::vFloat angle_lo = sfpi::dst_reg[angle_lo_base + i];
        sfpi::vFloat quotient = angle_hi * inverse_two_pi + rounding_bias_value;
        quotient = quotient - rounding_bias_value;
        sfpi::vFloat period_hi = quotient * negative_two_pi_hi;
        sfpi::vFloat period_lo = quotient * negative_two_pi_head - period_hi;
        period_lo = period_lo + quotient * negative_two_pi_tail;
        sfpi::vFloat reduced_hi = angle_hi + period_hi;
        sfpi::vFloat reduced_lo = angle_lo + period_lo;
        reduced_lo = reduced_lo + quotient * negative_two_pi_lo;
        sfpi::dst_reg[out_base + i] = reduced_hi + reduced_lo;
    }
}

inline void ordinary_reduce_face(uint32_t angle_tile, uint32_t, uint32_t out) {
    constexpr float inverse_two_pi = 0x1.45f306p-3f;
    constexpr float negative_two_pi_hi = -0x1.921fb6p+2f;
    constexpr float negative_two_pi_lo = 0x1.777a5cp-23f;
    constexpr float negative_two_pi_head = -0x1.922p+2f;
    constexpr float negative_two_pi_tail = 0x1.28p-16f;
    constexpr float rounding_bias_value = 0x1.8p23f;
    const uint32_t angle_base = angle_tile * kVectorsPerTile;
    const uint32_t out_base = out * kVectorsPerTile;
    for (std::size_t i = 0; i < kVectorsPerFace; ++i) {
        sfpi::vFloat angle = sfpi::dst_reg[angle_base + i];
        sfpi::vFloat quotient = angle * inverse_two_pi + rounding_bias_value;
        quotient = quotient - rounding_bias_value;
        sfpi::vFloat period_hi = quotient * negative_two_pi_hi;
        sfpi::vFloat period_lo = quotient * negative_two_pi_head - period_hi;
        period_lo = period_lo + quotient * negative_two_pi_tail;
        sfpi::vFloat reduced = angle + period_hi;
        reduced = reduced + period_lo + quotient * negative_two_pi_lo;
        sfpi::dst_reg[out_base + i] = reduced;
    }
}

inline void safe_denominator_face(uint32_t value, uint32_t zero_mask, uint32_t out) {
    const uint32_t value_base = value * kVectorsPerTile;
    const uint32_t mask_base = zero_mask * kVectorsPerTile;
    const uint32_t out_base = out * kVectorsPerTile;
    for (std::size_t i = 0; i < kVectorsPerFace; ++i) {
        sfpi::vFloat selected = sfpi::dst_reg[value_base + i];
        sfpi::vFloat mask = sfpi::dst_reg[mask_base + i];
        v_if(mask != 0.0f) { selected = 1.0f; }
        v_endif;
        sfpi::dst_reg[out_base + i] = selected;
    }
}

inline void zero_at_mask_face(uint32_t value, uint32_t zero_mask, uint32_t out) {
    const uint32_t value_base = value * kVectorsPerTile;
    const uint32_t mask_base = zero_mask * kVectorsPerTile;
    const uint32_t out_base = out * kVectorsPerTile;
    for (std::size_t i = 0; i < kVectorsPerFace; ++i) {
        sfpi::vFloat selected = sfpi::dst_reg[value_base + i];
        sfpi::vFloat mask = sfpi::dst_reg[mask_base + i];
        v_if(mask != 0.0f) { selected = 0.0f; }
        v_endif;
        sfpi::dst_reg[out_base + i] = selected;
    }
}

inline void one_at_mask_face(uint32_t value, uint32_t zero_mask, uint32_t out) {
    const uint32_t value_base = value * kVectorsPerTile;
    const uint32_t mask_base = zero_mask * kVectorsPerTile;
    const uint32_t out_base = out * kVectorsPerTile;
    for (std::size_t i = 0; i < kVectorsPerFace; ++i) {
        sfpi::vFloat selected = sfpi::dst_reg[value_base + i];
        sfpi::vFloat mask = sfpi::dst_reg[mask_base + i];
        v_if(mask != 0.0f) { selected = 1.0f; }
        v_endif;
        sfpi::dst_reg[out_base + i] = selected;
    }
}

inline void negate_face(uint32_t value, uint32_t, uint32_t out) {
    const uint32_t value_base = value * kVectorsPerTile;
    const uint32_t out_base = out * kVectorsPerTile;
    for (std::size_t i = 0; i < kVectorsPerFace; ++i) {
        sfpi::vFloat source = sfpi::dst_reg[value_base + i];
        sfpi::dst_reg[out_base + i] = -source;
    }
}

}  // namespace ckernel::sfpu
