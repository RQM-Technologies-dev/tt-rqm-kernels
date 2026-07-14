// SPDX-FileCopyrightText: 2026 RQM Technologies LLC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

#include "sfpi.h"

namespace ckernel::sfpu {

// Dst tiles 0..3 contain a=(w,x,y,z), and 4..7 contain b=(w,x,y,z).
// Each function covers one tile face and writes its result to Dst tile 0.
// Keeping only one result live avoids exceeding the Wormhole SFPI register
// allocator while leaving all Hamilton-product arithmetic on SFPU.
inline void qmul_w_tile_face() {
    constexpr std::size_t vectors_per_face = 8;
    constexpr std::size_t vectors_per_tile = 32;

    for (std::size_t i = 0; i < vectors_per_face; ++i) {
        sfpi::vFloat result = sfpi::dst_reg[0 * vectors_per_tile + i] * sfpi::dst_reg[4 * vectors_per_tile + i];
        result = result - sfpi::dst_reg[1 * vectors_per_tile + i] * sfpi::dst_reg[5 * vectors_per_tile + i];
        result = result - sfpi::dst_reg[2 * vectors_per_tile + i] * sfpi::dst_reg[6 * vectors_per_tile + i];
        result = result - sfpi::dst_reg[3 * vectors_per_tile + i] * sfpi::dst_reg[7 * vectors_per_tile + i];
        sfpi::dst_reg[i] = result;
    }
}

inline void qmul_x_tile_face() {
    constexpr std::size_t vectors_per_face = 8;
    constexpr std::size_t vectors_per_tile = 32;
    for (std::size_t i = 0; i < vectors_per_face; ++i) {
        sfpi::vFloat result = sfpi::dst_reg[0 * vectors_per_tile + i] * sfpi::dst_reg[5 * vectors_per_tile + i];
        result = result + sfpi::dst_reg[1 * vectors_per_tile + i] * sfpi::dst_reg[4 * vectors_per_tile + i];
        result = result + sfpi::dst_reg[2 * vectors_per_tile + i] * sfpi::dst_reg[7 * vectors_per_tile + i];
        result = result - sfpi::dst_reg[3 * vectors_per_tile + i] * sfpi::dst_reg[6 * vectors_per_tile + i];
        sfpi::dst_reg[i] = result;
    }
}

inline void qmul_y_tile_face() {
    constexpr std::size_t vectors_per_face = 8;
    constexpr std::size_t vectors_per_tile = 32;
    for (std::size_t i = 0; i < vectors_per_face; ++i) {
        sfpi::vFloat result = sfpi::dst_reg[0 * vectors_per_tile + i] * sfpi::dst_reg[6 * vectors_per_tile + i];
        result = result - sfpi::dst_reg[1 * vectors_per_tile + i] * sfpi::dst_reg[7 * vectors_per_tile + i];
        result = result + sfpi::dst_reg[2 * vectors_per_tile + i] * sfpi::dst_reg[4 * vectors_per_tile + i];
        result = result + sfpi::dst_reg[3 * vectors_per_tile + i] * sfpi::dst_reg[5 * vectors_per_tile + i];
        sfpi::dst_reg[i] = result;
    }
}

inline void qmul_z_tile_face() {
    constexpr std::size_t vectors_per_face = 8;
    constexpr std::size_t vectors_per_tile = 32;
    for (std::size_t i = 0; i < vectors_per_face; ++i) {
        sfpi::vFloat result = sfpi::dst_reg[0 * vectors_per_tile + i] * sfpi::dst_reg[7 * vectors_per_tile + i];
        result = result + sfpi::dst_reg[1 * vectors_per_tile + i] * sfpi::dst_reg[6 * vectors_per_tile + i];
        result = result - sfpi::dst_reg[2 * vectors_per_tile + i] * sfpi::dst_reg[5 * vectors_per_tile + i];
        result = result + sfpi::dst_reg[3 * vectors_per_tile + i] * sfpi::dst_reg[4 * vectors_per_tile + i];
        sfpi::dst_reg[i] = result;
    }
}

}  // namespace ckernel::sfpu
