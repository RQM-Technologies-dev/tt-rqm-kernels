// SPDX-FileCopyrightText: 2026 RQM Technologies LLC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/comp.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/eltwise_unary/trigonometry.h"
#include "api/compute/tile_move_copy.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"

#ifdef TRISC_MATH
#include "h2a_sfpu.h"
#endif

inline void primitive_safe_select() {
    MATH(SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE, DST_ACCUM_MODE, h2a_safe_denominator_tile_face, 0, 1, 0, VectorMode::RC));
}

void kernel_main() {
    const uint32_t tile_count = get_arg_val<uint32_t>(0);
    init_sfpu(0, 16);
    for (uint32_t tile = 0; tile < tile_count; ++tile) {
        for (uint32_t cb = 0; cb < 6; ++cb) cb_wait_front(cb, 1);
        tile_regs_acquire();
        copy_tile(5, 0, 0);  // The protocol's positive 1/hbar plane.
        sqrt_tile_init();
        sqrt_tile<false>(0);
        recip_tile_init<false>();
        recip_tile<false>(0);
        sin_tile_init();
        sin_tile(0);
        cos_tile_init();
        cos_tile(0);
        copy_tile(1, 0, 1);
        eqz_tile_init();
        eqz_tile(1);
        primitive_safe_select();
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t cb = 16; cb < 22; ++cb) {
            cb_reserve_back(cb, 1);
            pack_tile(0, cb);
            cb_push_back(cb, 1);
        }
        tile_regs_release();
        for (uint32_t cb = 0; cb < 6; ++cb) cb_pop_front(cb, 1);
    }
}
