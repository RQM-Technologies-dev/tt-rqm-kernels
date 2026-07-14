// SPDX-FileCopyrightText: 2026 RQM Technologies LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"
#include "llk_math_eltwise_ternary_sfpu_macros.h"

#ifdef TRISC_MATH
#include "su2_sfpu.h"
#endif

inline void su2_product_sfpu() {
    MATH(SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE, DST_ACCUM_MODE, su2_product_tile_face, 0, 1, 0, VectorMode::RC));
}

inline void su2_add_product_sfpu() {
    MATH(SFPU_TERNARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE, DST_ACCUM_MODE, su2_add_product_tile_face, 0, 1, 2, 0, VectorMode::RC));
}

inline void su2_subtract_product_sfpu() {
    MATH(SFPU_TERNARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE, DST_ACCUM_MODE, su2_subtract_product_tile_face, 0, 1, 2, 0, VectorMode::RC));
}

inline void su2_begin_product(uint32_t lhs_cb, uint32_t rhs_cb) {
    copy_tile(lhs_cb, 0, 0);
    copy_tile(rhs_cb, 0, 1);
    su2_product_sfpu();
}

inline void su2_accumulate_product(uint32_t lhs_cb, uint32_t rhs_cb, bool subtract) {
    copy_tile(lhs_cb, 0, 1);
    copy_tile(rhs_cb, 0, 2);
    subtract ? su2_subtract_product_sfpu() : su2_add_product_sfpu();
}

inline void su2_quaternion_component(uint32_t lhs, uint32_t rhs, uint32_t lane) {
    if (lane == 0) {
        su2_begin_product(lhs + 0, rhs + 0);
        su2_accumulate_product(lhs + 1, rhs + 1, true);
        su2_accumulate_product(lhs + 2, rhs + 2, true);
        su2_accumulate_product(lhs + 3, rhs + 3, true);
    } else if (lane == 1) {
        su2_begin_product(lhs + 0, rhs + 1);
        su2_accumulate_product(lhs + 1, rhs + 0, false);
        su2_accumulate_product(lhs + 2, rhs + 3, false);
        su2_accumulate_product(lhs + 3, rhs + 2, true);
    } else if (lane == 2) {
        su2_begin_product(lhs + 0, rhs + 2);
        su2_accumulate_product(lhs + 1, rhs + 3, true);
        su2_accumulate_product(lhs + 2, rhs + 0, false);
        su2_accumulate_product(lhs + 3, rhs + 1, false);
    } else {
        su2_begin_product(lhs + 0, rhs + 3);
        su2_accumulate_product(lhs + 1, rhs + 2, false);
        su2_accumulate_product(lhs + 2, rhs + 1, true);
        su2_accumulate_product(lhs + 3, rhs + 0, false);
    }
}

inline void su2_phase_component(uint32_t lhs, uint32_t rhs, uint32_t lane) {
    if (lane == 0) {
        su2_begin_product(lhs + 4, rhs + 4);
        su2_accumulate_product(lhs + 5, rhs + 5, true);
    } else {
        su2_begin_product(lhs + 4, rhs + 5);
        su2_accumulate_product(lhs + 5, rhs + 4, false);
    }
}

inline void su2_pack_component(uint32_t output_cb) {
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(output_cb, 1);
    pack_tile(0, output_cb);
    cb_push_back(output_cb, 1);
    tile_regs_release();
}

inline void su2_calculate_and_pack(uint32_t lhs, uint32_t rhs, uint32_t output, uint32_t lane) {
    tile_regs_acquire();
    if (lane < 4) su2_quaternion_component(lhs, rhs, lane);
    else su2_phase_component(lhs, rhs, lane - 4);
    su2_pack_component(output + lane);
}
