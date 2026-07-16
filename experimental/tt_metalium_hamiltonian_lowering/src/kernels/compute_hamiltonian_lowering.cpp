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

inline void h2a_product() {
    MATH(SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE, DST_ACCUM_MODE, h2a_product_tile_face, 0, 1, 0, VectorMode::RC));
}

inline void h2a_add() {
    MATH(SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE, DST_ACCUM_MODE, h2a_add_tile_face, 0, 1, 0, VectorMode::RC));
}

inline void h2a_safe_denominator() {
    MATH(SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE, DST_ACCUM_MODE, h2a_safe_denominator_tile_face, 0, 1, 0, VectorMode::RC));
}

inline void h2a_zero_at_mask() {
    MATH(SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE, DST_ACCUM_MODE, h2a_zero_at_mask_tile_face, 0, 1, 0, VectorMode::RC));
}

inline void h2a_one_at_mask() {
    MATH(SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE, DST_ACCUM_MODE, h2a_one_at_mask_tile_face, 0, 1, 0, VectorMode::RC));
}

inline void h2a_negate() {
    MATH(SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE, DST_ACCUM_MODE, h2a_negate_tile_face, 0, 1, 0, VectorMode::RC));
}

inline void h2a_pack(uint32_t cb) {
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb, 1);
    pack_tile(0, cb);
    cb_push_back(cb, 1);
    tile_regs_release();
}

inline void h2a_binary(uint32_t lhs, uint32_t rhs, uint32_t out, bool add) {
    cb_wait_front(lhs, 1);
    cb_wait_front(rhs, 1);
    tile_regs_acquire();
    copy_tile(lhs, 0, 0);
    copy_tile(rhs, 0, 1);
    add ? h2a_add() : h2a_product();
    h2a_pack(out);
}

inline void h2a_unary(uint32_t input, uint32_t out, uint32_t operation) {
    cb_wait_front(input, 1);
    tile_regs_acquire();
    copy_tile(input, 0, 0);
    if (operation == 0) {
        sqrt_tile_init();
        sqrt_tile<false>(0);
    } else if (operation == 1) {
        recip_tile_init<false>();
        recip_tile<false>(0);
    } else if (operation == 2) {
        sin_tile_init();
        sin_tile(0);
    } else if (operation == 3) {
        cos_tile_init();
        cos_tile(0);
    } else {
        eqz_tile_init();
        eqz_tile(0);
    }
    h2a_pack(out);
}

inline void h2a_select(uint32_t value, uint32_t mask, uint32_t out, uint32_t mode) {
    cb_wait_front(value, 1);
    cb_wait_front(mask, 1);
    tile_regs_acquire();
    copy_tile(value, 0, 0);
    copy_tile(value, 0, 1);
    copy_tile(mask, 0, 1);
    if (mode == 0) h2a_safe_denominator();
    else if (mode == 1) h2a_zero_at_mask();
    else h2a_one_at_mask();
    h2a_pack(out);
}

inline void h2a_negated(uint32_t value, uint32_t out) {
    cb_wait_front(value, 1);
    tile_regs_acquire();
    copy_tile(value, 0, 0);
    h2a_negate();
    h2a_pack(out);
}

inline void h2a_pop(uint32_t cb) { cb_pop_front(cb, 1); }

void kernel_main() {
    const uint32_t tile_count = get_arg_val<uint32_t>(0);
    constexpr uint32_t h0 = 0, hx = 1, hy = 2, hz = 3, dt = 4, inv_hbar = 5;
    constexpr uint32_t step = 6, hx2 = 7, hy2 = 8, hz2 = 9, sum_xy = 10, r2 = 11;
    constexpr uint32_t zero_mask = 12, r = 13, safe_r = 14, inv_r = 15;
    constexpr uint32_t out_w = 16, out_x = 17, out_y = 18, out_z = 19, out_pr = 20, out_pi = 21;
    constexpr uint32_t theta = 22, alpha = 23, sin_theta = 24, cos_theta = 25;
    constexpr uint32_t vector_scale = 26, sin_alpha = 27, cos_alpha = 28;
    constexpr uint32_t temp_x = 29, temp_y = 30, temp_z = 31;
    init_sfpu(h0, out_w);

    for (uint32_t tile = 0; tile < tile_count; ++tile) {
        for (uint32_t cb = h0; cb <= inv_hbar; ++cb) cb_wait_front(cb, 1);

        h2a_binary(dt, inv_hbar, step, false);
        h2a_binary(hx, hx, hx2, false);
        h2a_binary(hy, hy, hy2, false);
        h2a_binary(hz, hz, hz2, false);
        h2a_binary(hx2, hy2, sum_xy, true);
        h2a_pop(hx2); h2a_pop(hy2);
        h2a_binary(sum_xy, hz2, r2, true);
        h2a_pop(sum_xy); h2a_pop(hz2);
        h2a_unary(r2, zero_mask, 4);
        h2a_unary(r2, r, 0);
        h2a_pop(r2);
        h2a_select(r, zero_mask, safe_r, 0);
        h2a_unary(safe_r, inv_r, 1);
        h2a_pop(safe_r);

        h2a_binary(r, step, theta, false);
        h2a_pop(r);
        h2a_binary(h0, step, alpha, false);
        h2a_pop(step);
        h2a_unary(theta, sin_theta, 2);
        h2a_unary(theta, cos_theta, 3);
        h2a_pop(theta);
        h2a_binary(sin_theta, inv_r, vector_scale, false);
        h2a_pop(sin_theta); h2a_pop(inv_r);

        h2a_binary(hx, vector_scale, temp_x, false);
        h2a_select(temp_x, zero_mask, out_x, 1); h2a_pop(temp_x);
        h2a_binary(hy, vector_scale, temp_y, false);
        h2a_select(temp_y, zero_mask, out_y, 1); h2a_pop(temp_y);
        h2a_binary(hz, vector_scale, temp_z, false);
        h2a_select(temp_z, zero_mask, out_z, 1); h2a_pop(temp_z);
        h2a_pop(vector_scale);
        h2a_select(cos_theta, zero_mask, out_w, 2);
        h2a_pop(cos_theta); h2a_pop(zero_mask);

        h2a_unary(alpha, sin_alpha, 2);
        h2a_unary(alpha, cos_alpha, 3);
        h2a_pop(alpha);
        h2a_negated(sin_alpha, out_pi);
        h2a_pop(sin_alpha);
        cb_wait_front(cos_alpha, 1);
        tile_regs_acquire(); copy_tile(cos_alpha, 0, 0); h2a_pack(out_pr);
        h2a_pop(cos_alpha);

        for (uint32_t cb = h0; cb <= inv_hbar; ++cb) h2a_pop(cb);
    }
}
