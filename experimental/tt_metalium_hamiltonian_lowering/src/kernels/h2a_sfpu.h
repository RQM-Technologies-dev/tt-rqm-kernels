// SPDX-FileCopyrightText: 2026 RQM Technologies LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

#include "sfpi.h"

namespace ckernel::sfpu {

constexpr std::size_t kH2aVectorsPerFace = 8;
constexpr std::size_t kH2aVectorsPerTile = 32;

inline void h2a_product_tile_face(uint32_t lhs, uint32_t rhs, uint32_t out) {
    const uint32_t lhs_base = lhs * kH2aVectorsPerTile;
    const uint32_t rhs_base = rhs * kH2aVectorsPerTile;
    const uint32_t out_base = out * kH2aVectorsPerTile;
    for (std::size_t i = 0; i < kH2aVectorsPerFace; ++i) {
        sfpi::dst_reg[out_base + i] = sfpi::dst_reg[lhs_base + i] * sfpi::dst_reg[rhs_base + i];
    }
}

inline void h2a_add_tile_face(uint32_t lhs, uint32_t rhs, uint32_t out) {
    const uint32_t lhs_base = lhs * kH2aVectorsPerTile;
    const uint32_t rhs_base = rhs * kH2aVectorsPerTile;
    const uint32_t out_base = out * kH2aVectorsPerTile;
    for (std::size_t i = 0; i < kH2aVectorsPerFace; ++i) {
        sfpi::vFloat lhs_value = sfpi::dst_reg[lhs_base + i];
        sfpi::vFloat rhs_value = sfpi::dst_reg[rhs_base + i];
        sfpi::dst_reg[out_base + i] = lhs_value + rhs_value;
    }
}

inline void h2a_safe_denominator_tile_face(uint32_t value, uint32_t zero_mask, uint32_t out) {
    const uint32_t value_base = value * kH2aVectorsPerTile;
    const uint32_t mask_base = zero_mask * kH2aVectorsPerTile;
    const uint32_t out_base = out * kH2aVectorsPerTile;
    for (std::size_t i = 0; i < kH2aVectorsPerFace; ++i) {
        sfpi::vFloat selected = sfpi::dst_reg[value_base + i];
        sfpi::vFloat mask = sfpi::dst_reg[mask_base + i];
        v_if(mask != 0.0f) { selected = 1.0f; }
        v_endif;
        sfpi::dst_reg[out_base + i] = selected;
    }
}

inline void h2a_zero_at_mask_tile_face(uint32_t value, uint32_t zero_mask, uint32_t out) {
    const uint32_t value_base = value * kH2aVectorsPerTile;
    const uint32_t mask_base = zero_mask * kH2aVectorsPerTile;
    const uint32_t out_base = out * kH2aVectorsPerTile;
    for (std::size_t i = 0; i < kH2aVectorsPerFace; ++i) {
        sfpi::vFloat selected = sfpi::dst_reg[value_base + i];
        sfpi::vFloat mask = sfpi::dst_reg[mask_base + i];
        v_if(mask != 0.0f) { selected = 0.0f; }
        v_endif;
        sfpi::dst_reg[out_base + i] = selected;
    }
}

inline void h2a_one_at_mask_tile_face(uint32_t value, uint32_t zero_mask, uint32_t out) {
    const uint32_t value_base = value * kH2aVectorsPerTile;
    const uint32_t mask_base = zero_mask * kH2aVectorsPerTile;
    const uint32_t out_base = out * kH2aVectorsPerTile;
    for (std::size_t i = 0; i < kH2aVectorsPerFace; ++i) {
        sfpi::vFloat selected = sfpi::dst_reg[value_base + i];
        sfpi::vFloat mask = sfpi::dst_reg[mask_base + i];
        v_if(mask != 0.0f) { selected = 1.0f; }
        v_endif;
        sfpi::dst_reg[out_base + i] = selected;
    }
}

inline void h2a_negate_tile_face(uint32_t value, uint32_t, uint32_t out) {
    const uint32_t value_base = value * kH2aVectorsPerTile;
    const uint32_t out_base = out * kH2aVectorsPerTile;
    for (std::size_t i = 0; i < kH2aVectorsPerFace; ++i) {
        sfpi::vFloat source = sfpi::dst_reg[value_base + i];
        sfpi::dst_reg[out_base + i] = -source;
    }
}

}  // namespace ckernel::sfpu
