// SPDX-FileCopyrightText: 2026 RQM Technologies LLC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

#include "sfpi.h"

namespace ckernel::sfpu {

// Dst tiles 0..3 contain a=(w,x,y,z), and 4..7 contain b=(w,x,y,z).
// Each invocation covers one tile face. All Hamilton-product arithmetic is
// performed by SFPU vector instructions before results overwrite Dst 0..3.
inline void qmul_tile_face() {
    constexpr std::size_t vectors_per_face = 8;
    constexpr std::size_t vectors_per_tile = 32;

    for (std::size_t i = 0; i < vectors_per_face; ++i) {
        sfpi::vFloat aw = sfpi::dst_reg[0 * vectors_per_tile + i];
        sfpi::vFloat ax = sfpi::dst_reg[1 * vectors_per_tile + i];
        sfpi::vFloat ay = sfpi::dst_reg[2 * vectors_per_tile + i];
        sfpi::vFloat az = sfpi::dst_reg[3 * vectors_per_tile + i];
        sfpi::vFloat bw = sfpi::dst_reg[4 * vectors_per_tile + i];
        sfpi::vFloat bx = sfpi::dst_reg[5 * vectors_per_tile + i];
        sfpi::vFloat by = sfpi::dst_reg[6 * vectors_per_tile + i];
        sfpi::vFloat bz = sfpi::dst_reg[7 * vectors_per_tile + i];

        sfpi::vFloat out_w = aw * bw - ax * bx - ay * by - az * bz;
        sfpi::vFloat out_x = aw * bx + ax * bw + ay * bz - az * by;
        sfpi::vFloat out_y = aw * by - ax * bz + ay * bw + az * bx;
        sfpi::vFloat out_z = aw * bz + ax * by - ay * bx + az * bw;

        sfpi::dst_reg[0 * vectors_per_tile + i] = out_w;
        sfpi::dst_reg[1 * vectors_per_tile + i] = out_x;
        sfpi::dst_reg[2 * vectors_per_tile + i] = out_y;
        sfpi::dst_reg[3 * vectors_per_tile + i] = out_z;
    }
}

}  // namespace ckernel::sfpu
