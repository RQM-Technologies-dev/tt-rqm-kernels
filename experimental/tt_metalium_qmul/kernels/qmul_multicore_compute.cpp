// SPDX-FileCopyrightText: 2026 RQM Technologies LLC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"

#ifdef TRISC_MATH
#include "qmul_sfpu.h"
#endif

#define TT_RQM_QMUL_SFPU_CALL(component)                                                        \
    MATH((_llk_math_eltwise_unary_sfpu_params_(                                                 \
        ::ckernel::sfpu::qmul_##component##_tile_face, 0, VectorMode::RC)))

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
            for (uint32_t input_lane = 0; input_lane < 8; ++input_lane) {
                copy_tile(first_input_cb + input_lane, 0, input_lane);
            }
            if (output_lane == 0) {
                TT_RQM_QMUL_SFPU_CALL(w);
            } else if (output_lane == 1) {
                TT_RQM_QMUL_SFPU_CALL(x);
            } else if (output_lane == 2) {
                TT_RQM_QMUL_SFPU_CALL(y);
            } else {
                TT_RQM_QMUL_SFPU_CALL(z);
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
