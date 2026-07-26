// SPDX-FileCopyrightText: 2026 RQM Technologies LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/trigonometry.h"
#include "api/compute/tile_move_copy.h"
#include "llk_math_eltwise_ternary_sfpu_macros.h"

#ifdef TRISC_MATH
#include "su4q_sfpu.h"
#endif

inline void su4q_product() {
    MATH(SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE, DST_ACCUM_MODE, su4q_product_face, 0, 1, 0, VectorMode::RC));
}

inline void su4q_add_product() {
    MATH(SFPU_TERNARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE, DST_ACCUM_MODE, su4q_add_product_face, 0, 1, 2, 0, VectorMode::RC));
}

inline void su4q_subtract_product() {
    MATH(SFPU_TERNARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        su4q_subtract_product_face,
        0,
        1,
        2,
        0,
        VectorMode::RC));
}

inline void su4q_pack_one(uint32_t cb) {
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb, 1);
    pack_tile(0, cb);
    cb_push_back(cb, 1);
    tile_regs_release();
}

inline void su4q_first_product(uint32_t lhs_cb, uint32_t rhs_cb) {
    copy_tile(lhs_cb, 0, 0);
    copy_tile(rhs_cb, 0, 1);
    su4q_product();
}

inline void su4q_accumulate_product(
    uint32_t lhs_cb, uint32_t rhs_cb, bool subtract) {
    copy_tile(lhs_cb, 0, 1);
    copy_tile(rhs_cb, 0, 2);
    subtract ? su4q_subtract_product() : su4q_add_product();
}

inline void su4q_trig(uint32_t input_cb, uint32_t output_cb, bool sine) {
    cb_wait_front(input_cb, 1);
    tile_regs_acquire();
    copy_tile(input_cb, 0, 0);
    if (sine) {
        sin_tile_init();
        sin_tile(0);
    } else {
        cos_tile_init();
        cos_tile(0);
    }
    su4q_pack_one(output_cb);
}
