// SPDX-FileCopyrightText: 2026 RQM Technologies LLC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

#include "sfpi.h"

namespace ckernel::sfpu {

constexpr std::size_t kVectorsPerFace = 8;
constexpr std::size_t kVectorsPerTile = 32;

inline void qmul_product_tile_face(
    const uint32_t lhs_index,
    const uint32_t rhs_index,
    const uint32_t out_index) {
    const uint32_t lhs_base = lhs_index * kVectorsPerTile;
    const uint32_t rhs_base = rhs_index * kVectorsPerTile;
    const uint32_t out_base = out_index * kVectorsPerTile;
    for (std::size_t i = 0; i < kVectorsPerFace; ++i) {
        sfpi::dst_reg[out_base + i] = sfpi::dst_reg[lhs_base + i] * sfpi::dst_reg[rhs_base + i];
    }
}

inline void qmul_add_product_tile_face(
    const uint32_t accumulator_index,
    const uint32_t lhs_index,
    const uint32_t rhs_index,
    const uint32_t out_index) {
    const uint32_t accumulator_base = accumulator_index * kVectorsPerTile;
    const uint32_t lhs_base = lhs_index * kVectorsPerTile;
    const uint32_t rhs_base = rhs_index * kVectorsPerTile;
    const uint32_t out_base = out_index * kVectorsPerTile;
    for (std::size_t i = 0; i < kVectorsPerFace; ++i) {
        sfpi::dst_reg[out_base + i] =
            sfpi::dst_reg[accumulator_base + i] + sfpi::dst_reg[lhs_base + i] * sfpi::dst_reg[rhs_base + i];
    }
}

inline void qmul_subtract_product_tile_face(
    const uint32_t accumulator_index,
    const uint32_t lhs_index,
    const uint32_t rhs_index,
    const uint32_t out_index) {
    const uint32_t accumulator_base = accumulator_index * kVectorsPerTile;
    const uint32_t lhs_base = lhs_index * kVectorsPerTile;
    const uint32_t rhs_base = rhs_index * kVectorsPerTile;
    const uint32_t out_base = out_index * kVectorsPerTile;
    for (std::size_t i = 0; i < kVectorsPerFace; ++i) {
        sfpi::dst_reg[out_base + i] =
            sfpi::dst_reg[accumulator_base + i] - sfpi::dst_reg[lhs_base + i] * sfpi::dst_reg[rhs_base + i];
    }
}

}  // namespace ckernel::sfpu
