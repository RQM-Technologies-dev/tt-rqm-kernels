// SPDX-FileCopyrightText: 2026 RQM Technologies LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include "sfpi.h"

namespace ckernel::sfpu {

constexpr std::size_t kSu4qVectorsPerFace = 8;
constexpr std::size_t kSu4qVectorsPerTile = 32;

inline void su4q_product_face(uint32_t lhs, uint32_t rhs, uint32_t out) {
    const uint32_t lhs_base = lhs * kSu4qVectorsPerTile;
    const uint32_t rhs_base = rhs * kSu4qVectorsPerTile;
    const uint32_t out_base = out * kSu4qVectorsPerTile;
    for (std::size_t index = 0; index < kSu4qVectorsPerFace; ++index) {
        sfpi::dst_reg[out_base + index] =
            sfpi::dst_reg[lhs_base + index] * sfpi::dst_reg[rhs_base + index];
    }
}

inline void su4q_add_product_face(
    uint32_t accumulator, uint32_t lhs, uint32_t rhs, uint32_t out) {
    const uint32_t accumulator_base = accumulator * kSu4qVectorsPerTile;
    const uint32_t lhs_base = lhs * kSu4qVectorsPerTile;
    const uint32_t rhs_base = rhs * kSu4qVectorsPerTile;
    const uint32_t out_base = out * kSu4qVectorsPerTile;
    for (std::size_t index = 0; index < kSu4qVectorsPerFace; ++index) {
        sfpi::dst_reg[out_base + index] =
            sfpi::dst_reg[accumulator_base + index] +
            sfpi::dst_reg[lhs_base + index] * sfpi::dst_reg[rhs_base + index];
    }
}

inline void su4q_subtract_product_face(
    uint32_t accumulator, uint32_t lhs, uint32_t rhs, uint32_t out) {
    const uint32_t accumulator_base = accumulator * kSu4qVectorsPerTile;
    const uint32_t lhs_base = lhs * kSu4qVectorsPerTile;
    const uint32_t rhs_base = rhs * kSu4qVectorsPerTile;
    const uint32_t out_base = out * kSu4qVectorsPerTile;
    for (std::size_t index = 0; index < kSu4qVectorsPerFace; ++index) {
        sfpi::dst_reg[out_base + index] =
            sfpi::dst_reg[accumulator_base + index] -
            sfpi::dst_reg[lhs_base + index] * sfpi::dst_reg[rhs_base + index];
    }
}

}  // namespace ckernel::sfpu
