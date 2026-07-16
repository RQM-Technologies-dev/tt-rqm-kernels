// SPDX-FileCopyrightText: 2026 RQM Technologies LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/eltwise_unary/trigonometry.h"
#include "api/compute/tile_move_copy.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
#include "llk_math_eltwise_ternary_sfpu_macros.h"

#ifdef TRISC_MATH
#include "h2a_compensated_sfpu.h"
#endif

inline void diagnostic_product() {
    MATH(SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, product_face, 0, 1, 0, VectorMode::RC));
}
inline void diagnostic_add() {
    MATH(SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, add_face, 0, 1, 0, VectorMode::RC));
}
inline void diagnostic_two_product() {
    MATH(SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE, DST_ACCUM_MODE, split_two_product_face, 0, 1, 0, VectorMode::RC));
}
inline void diagnostic_angle_product() {
    MATH(SFPU_TERNARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE, DST_ACCUM_MODE, compensated_product_face, 0, 1, 2, 0, VectorMode::RC));
}
inline void diagnostic_reduce() {
    MATH(SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE, DST_ACCUM_MODE, compensated_reduce_face, 0, 1, 0, VectorMode::RC));
}
inline void diagnostic_ordinary_reduce() {
    MATH(SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE, DST_ACCUM_MODE, ordinary_reduce_face, 0, 1, 0, VectorMode::RC));
}
inline void diagnostic_pack_one(uint32_t cb) {
    tile_regs_commit(); tile_regs_wait(); cb_reserve_back(cb, 1); pack_tile(0, cb); cb_push_back(cb, 1); tile_regs_release();
}
inline void diagnostic_pack_two(uint32_t hi, uint32_t lo) {
    tile_regs_commit(); tile_regs_wait();
    cb_reserve_back(hi, 1); pack_tile(0, hi); cb_push_back(hi, 1);
    cb_reserve_back(lo, 1); pack_tile(1, lo); cb_push_back(lo, 1);
    tile_regs_release();
}
inline void diagnostic_binary(uint32_t lhs, uint32_t rhs, uint32_t out, bool add) {
    cb_wait_front(lhs, 1); cb_wait_front(rhs, 1); tile_regs_acquire();
    copy_tile(lhs, 0, 0); copy_tile(rhs, 0, 1); add ? diagnostic_add() : diagnostic_product(); diagnostic_pack_one(out);
}
inline void diagnostic_pair(uint32_t coefficient, uint32_t step_hi, uint32_t step_lo, uint32_t hi, uint32_t lo) {
    cb_wait_front(coefficient, 1); cb_wait_front(step_hi, 1); cb_wait_front(step_lo, 1); tile_regs_acquire();
    copy_tile(coefficient, 0, 0); copy_tile(step_hi, 0, 1); copy_tile(step_lo, 0, 2);
    diagnostic_angle_product(); diagnostic_pack_two(hi, lo);
}
inline void diagnostic_reduce_pair(uint32_t hi, uint32_t lo, uint32_t out) {
    cb_wait_front(hi, 1); cb_wait_front(lo, 1); tile_regs_acquire();
    copy_tile(hi, 0, 0); copy_tile(lo, 0, 1); diagnostic_reduce(); diagnostic_pack_one(out);
}
inline void diagnostic_reduce_one(uint32_t input, uint32_t out) {
    cb_wait_front(input, 1); tile_regs_acquire();
    copy_tile(input, 0, 0); copy_tile(input, 0, 1); diagnostic_ordinary_reduce(); diagnostic_pack_one(out);
}
inline void diagnostic_sqrt(uint32_t input, uint32_t out) {
    cb_wait_front(input, 1); tile_regs_acquire(); copy_tile(input, 0, 0);
    sqrt_tile_init(); sqrt_tile<false>(0); diagnostic_pack_one(out);
}
inline void diagnostic_trig(uint32_t input, uint32_t out, bool cosine) {
    cb_wait_front(input, 1); tile_regs_acquire(); copy_tile(input, 0, 0);
    if (cosine) { cos_tile_init(); cos_tile(0); } else { sin_tile_init(); sin_tile(0); }
    diagnostic_pack_one(out);
}
inline void diagnostic_copy(uint32_t input, uint32_t output) {
    cb_wait_front(input, 1); tile_regs_acquire(); copy_tile(input, 0, 0); diagnostic_pack_one(output);
}
inline void diagnostic_pop(uint32_t cb) { cb_pop_front(cb, 1); }

void kernel_main() {
    const uint32_t tile_count = get_arg_val<uint32_t>(0);
    constexpr uint32_t h0=0, hx=1, hy=2, hz=3, dt=4, inv_hbar=5;
    constexpr uint32_t step_hi=6, step_lo=7, hx2=8, hy2=9, hz2=10, sum_xy=11, r2=12, r=13;
    constexpr uint32_t theta_hi=14, theta_lo=15, out0=16, out1=17, out2=18, out3=19, out4=20, out5=21;
    constexpr uint32_t alpha_hi=22, alpha_lo=23, reduced_theta=24, reduced_alpha=25;
    constexpr uint32_t ordinary_theta=26, ordinary_alpha=27, sin_theta=28, cos_theta=29, sin_alpha=30, cos_alpha=31;
    init_sfpu(h0, out0);
    for (uint32_t tile=0; tile<tile_count; ++tile) {
        for (uint32_t cb=h0; cb<=inv_hbar; ++cb) cb_wait_front(cb, 1);
        tile_regs_acquire(); copy_tile(dt,0,0); copy_tile(inv_hbar,0,1); diagnostic_two_product(); diagnostic_pack_two(step_hi,step_lo);
        diagnostic_binary(hx,hx,hx2,false); diagnostic_binary(hy,hy,hy2,false); diagnostic_binary(hz,hz,hz2,false);
        diagnostic_binary(hx2,hy2,sum_xy,true); diagnostic_pop(hx2); diagnostic_pop(hy2);
        diagnostic_binary(sum_xy,hz2,r2,true); diagnostic_pop(sum_xy); diagnostic_pop(hz2);
        diagnostic_sqrt(r2,r);
        diagnostic_binary(r,step_hi,ordinary_theta,false);
        diagnostic_binary(h0,step_hi,ordinary_alpha,false);
#ifdef H2A_ORIGINAL_TRIG_DIAGNOSTIC
        diagnostic_reduce_one(ordinary_theta,reduced_theta);
        diagnostic_reduce_one(ordinary_alpha,reduced_alpha);
#else
        diagnostic_pair(r,step_hi,step_lo,theta_hi,theta_lo);
        diagnostic_pair(h0,step_hi,step_lo,alpha_hi,alpha_lo);
        diagnostic_reduce_pair(theta_hi,theta_lo,reduced_theta);
        diagnostic_reduce_pair(alpha_hi,alpha_lo,reduced_alpha);
#endif
#ifdef H2A_VALUE_DIAGNOSTIC
        diagnostic_copy(r2,out0); diagnostic_copy(r,out1); diagnostic_copy(step_hi,out2);
        diagnostic_copy(ordinary_theta,out3); diagnostic_copy(ordinary_alpha,out4); diagnostic_copy(reduced_theta,out5);
#else
        diagnostic_trig(reduced_theta,sin_theta,false); diagnostic_trig(reduced_theta,cos_theta,true);
        diagnostic_trig(reduced_alpha,sin_alpha,false); diagnostic_trig(reduced_alpha,cos_alpha,true);
        diagnostic_copy(reduced_theta,out0); diagnostic_copy(reduced_alpha,out1);
        diagnostic_copy(sin_theta,out2); diagnostic_copy(cos_theta,out3);
        diagnostic_copy(sin_alpha,out4); diagnostic_copy(cos_alpha,out5);
#endif
        for (uint32_t cb=h0; cb<=inv_hbar; ++cb) diagnostic_pop(cb);
    }
}
