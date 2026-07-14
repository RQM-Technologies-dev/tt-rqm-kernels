// SPDX-FileCopyrightText: 2026 RQM Technologies LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

#include "sfpi.h"

namespace ckernel::sfpu {

constexpr std::size_t kSu2VectorsPerFace = 8;
constexpr std::size_t kSu2VectorsPerTile = 32;

inline void su2_product_tile_face(uint32_t lhs, uint32_t rhs, uint32_t out) {
    const uint32_t lhs_base = lhs * kSu2VectorsPerTile;
    const uint32_t rhs_base = rhs * kSu2VectorsPerTile;
    const uint32_t out_base = out * kSu2VectorsPerTile;
    for (std::size_t i = 0; i < kSu2VectorsPerFace; ++i) {
        sfpi::dst_reg[out_base + i] = sfpi::dst_reg[lhs_base + i] * sfpi::dst_reg[rhs_base + i];
    }
}

inline void su2_add_product_tile_face(uint32_t accumulator, uint32_t lhs, uint32_t rhs, uint32_t out) {
    const uint32_t accumulator_base = accumulator * kSu2VectorsPerTile;
    const uint32_t lhs_base = lhs * kSu2VectorsPerTile;
    const uint32_t rhs_base = rhs * kSu2VectorsPerTile;
    const uint32_t out_base = out * kSu2VectorsPerTile;
    for (std::size_t i = 0; i < kSu2VectorsPerFace; ++i) {
        sfpi::dst_reg[out_base + i] =
            sfpi::dst_reg[accumulator_base + i] + sfpi::dst_reg[lhs_base + i] * sfpi::dst_reg[rhs_base + i];
    }
}

inline void su2_subtract_product_tile_face(uint32_t accumulator, uint32_t lhs, uint32_t rhs, uint32_t out) {
    const uint32_t accumulator_base = accumulator * kSu2VectorsPerTile;
    const uint32_t lhs_base = lhs * kSu2VectorsPerTile;
    const uint32_t rhs_base = rhs * kSu2VectorsPerTile;
    const uint32_t out_base = out * kSu2VectorsPerTile;
    for (std::size_t i = 0; i < kSu2VectorsPerFace; ++i) {
        sfpi::dst_reg[out_base + i] =
            sfpi::dst_reg[accumulator_base + i] - sfpi::dst_reg[lhs_base + i] * sfpi::dst_reg[rhs_base + i];
    }
}

}  // namespace ckernel::sfpu
