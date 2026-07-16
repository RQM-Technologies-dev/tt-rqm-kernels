// SPDX-FileCopyrightText: 2026 RQM Technologies LLC
// SPDX-License-Identifier: Apache-2.0

// Development-only six-plane diagnostic:
// step_hi, step_lo, theta_hi, theta_lo, alpha_hi, alpha_lo.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/tile_move_copy.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"

#ifdef TRISC_MATH
#include "h2a_compensated_sfpu.h"
#endif

inline void product() {
    MATH(SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, product_face, 0, 1, 0, VectorMode::RC));
}
inline void add() {
    MATH(SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, add_face, 0, 1, 0, VectorMode::RC));
}
inline void two_product() {
    MATH(SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE, DST_ACCUM_MODE, split_two_product_face, 0, 1, 0, VectorMode::RC));
}
inline void pack_one(uint32_t cb) {
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb, 1);
    pack_tile(0, cb);
    cb_push_back(cb, 1);
    tile_regs_release();
}
inline void pack_two(uint32_t hi_cb, uint32_t lo_cb) {
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(hi_cb, 1);
    pack_tile(0, hi_cb);
    cb_push_back(hi_cb, 1);
    cb_reserve_back(lo_cb, 1);
    pack_tile(1, lo_cb);
    cb_push_back(lo_cb, 1);
    tile_regs_release();
}
inline void binary(uint32_t lhs, uint32_t rhs, uint32_t out, bool use_add) {
    cb_wait_front(lhs, 1);
    cb_wait_front(rhs, 1);
    tile_regs_acquire();
    copy_tile(lhs, 0, 0);
    copy_tile(rhs, 0, 1);
    use_add ? add() : product();
    pack_one(out);
}
inline void compensated_product(uint32_t lhs, uint32_t rhs, uint32_t hi, uint32_t lo) {
    cb_wait_front(lhs, 1);
    cb_wait_front(rhs, 1);
    tile_regs_acquire();
    copy_tile(lhs, 0, 0);
    copy_tile(rhs, 0, 1);
    two_product();
    pack_two(hi, lo);
}
inline void square_root(uint32_t input, uint32_t out) {
    cb_wait_front(input, 1);
    tile_regs_acquire();
    copy_tile(input, 0, 0);
    sqrt_tile_init();
    sqrt_tile<false>(0);
    pack_one(out);
}
inline void copy_to_output(uint32_t input, uint32_t output) {
    cb_wait_front(input, 1);
    tile_regs_acquire();
    copy_tile(input, 0, 0);
    pack_one(output);
}
inline void pop(uint32_t cb) { cb_pop_front(cb, 1); }

void kernel_main() {
    const uint32_t tile_count = get_arg_val<uint32_t>(0);
    constexpr uint32_t h0 = 0, hx = 1, hy = 2, hz = 3, dt = 4, inv_hbar = 5;
    constexpr uint32_t step_hi = 6, step_lo = 7, hx2 = 8, hy2 = 9, hz2 = 10;
    constexpr uint32_t sum_xy = 11, r2 = 12, r = 13, theta_hi = 14, theta_lo = 15;
    constexpr uint32_t out0 = 16, out1 = 17, out2 = 18, out3 = 19, out4 = 20, out5 = 21;
    constexpr uint32_t alpha_hi = 22, alpha_lo = 23;
    init_sfpu(h0, out0);
    for (uint32_t tile = 0; tile < tile_count; ++tile) {
        for (uint32_t cb = h0; cb <= inv_hbar; ++cb) cb_wait_front(cb, 1);
        compensated_product(dt, inv_hbar, step_hi, step_lo);
        binary(hx, hx, hx2, false);
        binary(hy, hy, hy2, false);
        binary(hz, hz, hz2, false);
        binary(hx2, hy2, sum_xy, true);
        pop(hx2); pop(hy2);
        binary(sum_xy, hz2, r2, true);
        pop(sum_xy); pop(hz2);
        square_root(r2, r);
        pop(r2);
        compensated_product(r, step_hi, theta_hi, theta_lo);
        compensated_product(h0, step_hi, alpha_hi, alpha_lo);
        copy_to_output(step_hi, out0);
        copy_to_output(step_lo, out1);
        copy_to_output(theta_hi, out2);
        copy_to_output(theta_lo, out3);
        copy_to_output(alpha_hi, out4);
        copy_to_output(alpha_lo, out5);
        pop(step_hi); pop(step_lo); pop(r); pop(theta_hi); pop(theta_lo); pop(alpha_hi); pop(alpha_lo);
        for (uint32_t cb = h0; cb <= inv_hbar; ++cb) pop(cb);
    }
}
