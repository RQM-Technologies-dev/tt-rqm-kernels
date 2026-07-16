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
#include "llk_math_eltwise_ternary_sfpu_macros.h"

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
inline void compensated_angle_product() {
    MATH(SFPU_TERNARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE, DST_ACCUM_MODE, compensated_product_face, 0, 1, 2, 0, VectorMode::RC));
}
inline void compensated_reduce() {
    MATH(SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE, DST_ACCUM_MODE, compensated_reduce_face, 0, 1, 0, VectorMode::RC));
}
inline void safe_denominator() {
    MATH(SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE, DST_ACCUM_MODE, safe_denominator_face, 0, 1, 0, VectorMode::RC));
}
inline void zero_at_mask() {
    MATH(SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE, DST_ACCUM_MODE, zero_at_mask_face, 0, 1, 0, VectorMode::RC));
}
inline void one_at_mask() {
    MATH(SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE, DST_ACCUM_MODE, one_at_mask_face, 0, 1, 0, VectorMode::RC));
}
inline void negate() {
    MATH(SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, negate_face, 0, 1, 0, VectorMode::RC));
}

inline void pack_one(uint32_t cb) {
    tile_regs_commit(); tile_regs_wait(); cb_reserve_back(cb, 1); pack_tile(0, cb); cb_push_back(cb, 1); tile_regs_release();
}
inline void pack_two(uint32_t hi_cb, uint32_t lo_cb) {
    tile_regs_commit(); tile_regs_wait();
    cb_reserve_back(hi_cb, 1); pack_tile(0, hi_cb); cb_push_back(hi_cb, 1);
    cb_reserve_back(lo_cb, 1); pack_tile(1, lo_cb); cb_push_back(lo_cb, 1);
    tile_regs_release();
}
inline void binary(uint32_t lhs, uint32_t rhs, uint32_t out, bool use_add) {
    cb_wait_front(lhs, 1); cb_wait_front(rhs, 1); tile_regs_acquire();
    copy_tile(lhs, 0, 0); copy_tile(rhs, 0, 1); use_add ? add() : product(); pack_one(out);
}
inline void compensated_product(uint32_t lhs, uint32_t rhs, uint32_t hi, uint32_t lo) {
    cb_wait_front(lhs, 1); cb_wait_front(rhs, 1); tile_regs_acquire();
    copy_tile(lhs, 0, 0); copy_tile(rhs, 0, 1); two_product(); pack_two(hi, lo);
}
inline void angle_pair(uint32_t coefficient, uint32_t step_hi, uint32_t step_lo, uint32_t hi, uint32_t lo) {
    cb_wait_front(coefficient, 1); cb_wait_front(step_hi, 1); cb_wait_front(step_lo, 1); tile_regs_acquire();
    copy_tile(coefficient, 0, 0); copy_tile(step_hi, 0, 1); copy_tile(step_lo, 0, 2);
    compensated_angle_product(); pack_two(hi, lo);
}
inline void reduce_angle(uint32_t hi, uint32_t lo, uint32_t out) {
    cb_wait_front(hi, 1); cb_wait_front(lo, 1); tile_regs_acquire();
    copy_tile(hi, 0, 0); copy_tile(lo, 0, 1); compensated_reduce(); pack_one(out);
}
inline void unary(uint32_t input, uint32_t out, uint32_t operation) {
    cb_wait_front(input, 1); tile_regs_acquire(); copy_tile(input, 0, 0);
    if (operation == 0) { sqrt_tile_init(); sqrt_tile<false>(0); }
    else if (operation == 1) { recip_tile_init<false>(); recip_tile<false>(0); }
    else if (operation == 2) { sin_tile_init(); sin_tile(0); }
    else if (operation == 3) { cos_tile_init(); cos_tile(0); }
    else { eqz_tile_init(); eqz_tile(0); }
    pack_one(out);
}
inline void select(uint32_t value, uint32_t mask, uint32_t out, uint32_t mode) {
    cb_wait_front(value, 1); cb_wait_front(mask, 1); tile_regs_acquire(); copy_tile(value, 0, 0); copy_tile(mask, 0, 1);
    if (mode == 0) safe_denominator(); else if (mode == 1) zero_at_mask(); else one_at_mask();
    pack_one(out);
}
inline void negated(uint32_t value, uint32_t out) {
    cb_wait_front(value, 1); tile_regs_acquire(); copy_tile(value, 0, 0); copy_tile(value, 0, 1); negate(); pack_one(out);
}
inline void pop(uint32_t cb) { cb_pop_front(cb, 1); }

void kernel_main() {
    const uint32_t tile_count = get_arg_val<uint32_t>(0);
    constexpr uint32_t h0=0, hx=1, hy=2, hz=3, dt=4, inv_hbar=5;
    constexpr uint32_t step_hi=6, step_lo=7, hx2=8, hy2=9, hz2=10, sum_xy=11, r2=12, zero_mask=13, r=14, safe_r=15;
    constexpr uint32_t out_w=16, out_x=17, out_y=18, out_z=19, out_pr=20, out_pi=21;
    constexpr uint32_t inv_r=22, pair_hi=23, pair_lo=24, theta=25, alpha=26;
    constexpr uint32_t sin_theta=27, cos_theta=28, scale=29, temp=30, trig_temp=31;
    init_sfpu(h0, out_w);
    for (uint32_t tile=0; tile<tile_count; ++tile) {
        for (uint32_t cb=h0; cb<=inv_hbar; ++cb) cb_wait_front(cb, 1);
        compensated_product(dt, inv_hbar, step_hi, step_lo);
        binary(hx,hx,hx2,false); binary(hy,hy,hy2,false); binary(hz,hz,hz2,false);
        binary(hx2,hy2,sum_xy,true); pop(hx2); pop(hy2);
        binary(sum_xy,hz2,r2,true); pop(sum_xy); pop(hz2);
        unary(r2,zero_mask,4); unary(r2,r,0); pop(r2);
        select(r,zero_mask,safe_r,0); unary(safe_r,inv_r,1); pop(safe_r);
        angle_pair(r,step_hi,step_lo,pair_hi,pair_lo);
        reduce_angle(pair_hi,pair_lo,theta); pop(pair_hi); pop(pair_lo);
        angle_pair(h0,step_hi,step_lo,pair_hi,pair_lo);
        reduce_angle(pair_hi,pair_lo,alpha); pop(pair_hi); pop(pair_lo);
        pop(step_hi); pop(step_lo); pop(r);
        unary(theta,sin_theta,2); unary(theta,cos_theta,3); pop(theta);
        binary(sin_theta,inv_r,scale,false); pop(sin_theta); pop(inv_r);
        binary(hx,scale,temp,false); select(temp,zero_mask,out_x,1); pop(temp);
        binary(hy,scale,temp,false); select(temp,zero_mask,out_y,1); pop(temp);
        binary(hz,scale,temp,false); select(temp,zero_mask,out_z,1); pop(temp); pop(scale);
        select(cos_theta,zero_mask,out_w,2); pop(cos_theta); pop(zero_mask);
        unary(alpha,pair_hi,2); unary(alpha,trig_temp,3); pop(alpha);
        negated(pair_hi,out_pi); pop(pair_hi);
        cb_wait_front(trig_temp,1); tile_regs_acquire(); copy_tile(trig_temp,0,0); pack_one(out_pr); pop(trig_temp);
        for (uint32_t cb=h0; cb<=inv_hbar; ++cb) pop(cb);
    }
}
