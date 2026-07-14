// SPDX-FileCopyrightText: 2026 RQM Technologies LLC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_ternary_sfpu.h"
#include "api/compute/tile_move_copy.h"

#ifdef TRISC_MATH
#include "qmul_sfpu.h"
#endif

inline void qmul_product_sfpu() {
    MATH(SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE, DST_ACCUM_MODE, qmul_product_tile_face, 0, 1, 0, VectorMode::RC));
}

inline void qmul_add_product_sfpu() {
    MATH(SFPU_TERNARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE, DST_ACCUM_MODE, qmul_add_product_tile_face, 0, 1, 2, 0, VectorMode::RC));
}

inline void qmul_subtract_product_sfpu() {
    MATH(SFPU_TERNARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE, DST_ACCUM_MODE, qmul_subtract_product_tile_face, 0, 1, 2, 0, VectorMode::RC));
}

inline void qmul_component(
    uint32_t first_input_cb,
    uint32_t lhs_0,
    uint32_t rhs_0,
    uint32_t lhs_1,
    uint32_t rhs_1,
    bool subtract_1,
    uint32_t lhs_2,
    uint32_t rhs_2,
    bool subtract_2,
    uint32_t lhs_3,
    uint32_t rhs_3,
    bool subtract_3) {
    copy_tile(first_input_cb + lhs_0, 0, 0);
    copy_tile(first_input_cb + rhs_0, 0, 1);
    qmul_product_sfpu();

    copy_tile(first_input_cb + lhs_1, 0, 1);
    copy_tile(first_input_cb + rhs_1, 0, 2);
    subtract_1 ? qmul_subtract_product_sfpu() : qmul_add_product_sfpu();

    copy_tile(first_input_cb + lhs_2, 0, 1);
    copy_tile(first_input_cb + rhs_2, 0, 2);
    subtract_2 ? qmul_subtract_product_sfpu() : qmul_add_product_sfpu();

    copy_tile(first_input_cb + lhs_3, 0, 1);
    copy_tile(first_input_cb + rhs_3, 0, 2);
    subtract_3 ? qmul_subtract_product_sfpu() : qmul_add_product_sfpu();
}

void kernel_main() {
    const uint32_t tile_count = get_arg_val<uint32_t>(0);

    constexpr uint32_t first_input_cb = static_cast<uint32_t>(tt::CBIndex::c_0);
    constexpr uint32_t first_output_cb = static_cast<uint32_t>(tt::CBIndex::c_16);

    init_sfpu(first_input_cb, first_output_cb);

    for (uint32_t tile = 0; tile < tile_count; ++tile) {
        for (uint32_t lane = 0; lane < 8; ++lane) {
            cb_wait_front(first_input_cb + lane, 1);
        }

        for (uint32_t output_lane = 0; output_lane < 4; ++output_lane) {
            tile_regs_acquire();
            if (output_lane == 0) {
                qmul_component(first_input_cb, 0, 4, 1, 5, true, 2, 6, true, 3, 7, true);
            } else if (output_lane == 1) {
                qmul_component(first_input_cb, 0, 5, 1, 4, false, 2, 7, false, 3, 6, true);
            } else if (output_lane == 2) {
                qmul_component(first_input_cb, 0, 6, 1, 7, true, 2, 4, false, 3, 5, false);
            } else {
                qmul_component(first_input_cb, 0, 7, 1, 6, false, 2, 5, true, 3, 4, false);
            }
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(first_output_cb + output_lane, 1);
            pack_tile(0, first_output_cb + output_lane);
            cb_push_back(first_output_cb + output_lane, 1);
            tile_regs_release();
        }
        for (uint32_t lane = 0; lane < 8; ++lane) {
            cb_pop_front(first_input_cb + lane, 1);
        }
    }
}
