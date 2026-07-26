// SPDX-FileCopyrightText: 2026 RQM Technologies LLC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_common.h"

void kernel_main() {
    const uint32_t tile_count = get_arg_val<uint32_t>(0);
    const uint32_t pauli = get_arg_val<uint32_t>(1);
    constexpr uint32_t angle = 0;
    constexpr uint32_t state = 1;
    constexpr uint32_t sine = 9;
    constexpr uint32_t cosine = 10;
    constexpr uint32_t output = static_cast<uint32_t>(tt::CBIndex::c_16);
    init_sfpu(angle, output);
    for (uint32_t tile = 0; tile < tile_count; ++tile) {
        for (uint32_t cb = angle; cb < state + 8; ++cb) cb_wait_front(cb, 1);
        su4q_trig(angle, sine, true);
        su4q_trig(angle, cosine, false);
        cb_wait_front(sine, 1);
        cb_wait_front(cosine, 1);
        for (uint32_t lane = 0; lane < 8; ++lane) {
            const uint32_t amplitude = lane / 2;
            const bool imaginary = (lane & 1U) != 0;
            uint32_t partner = amplitude;
            bool positive = true;
            if (pauli == 0) {
                partner = amplitude ^ 3U;
            } else if (pauli == 1) {
                partner = amplitude ^ 3U;
                positive = amplitude == 1 || amplitude == 2;
            } else {
                positive = amplitude == 0 || amplitude == 3;
            }
            const uint32_t self_cb = state + lane;
            const uint32_t partner_cb = state + 2 * partner + (imaginary ? 0 : 1);
            tile_regs_acquire();
            su4q_first_product(cosine, self_cb);
            const bool subtract = imaginary ? !positive : positive;
            su4q_accumulate_product(sine, partner_cb, subtract);
            su4q_pack_one(output + lane);
        }
        cb_pop_front(sine, 1);
        cb_pop_front(cosine, 1);
        for (uint32_t cb = angle; cb < state + 8; ++cb) cb_pop_front(cb, 1);
    }
}
