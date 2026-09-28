// SPDX-FileCopyrightText: 2026 RQM Technologies LLC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_common.h"

void kernel_main() {
    const uint32_t tile_count = get_arg_val<uint32_t>(0);
    constexpr uint32_t phase = 0;
    constexpr uint32_t state = 1;
    constexpr uint32_t sine = 9;
    constexpr uint32_t cosine = 10;
    constexpr uint32_t output = static_cast<uint32_t>(tt::CBIndex::c_16);
    init_sfpu(phase, output);
    for (uint32_t tile = 0; tile < tile_count; ++tile) {
        for (uint32_t cb = phase; cb < state + 8; ++cb) cb_wait_front(cb, 1);
        su4q_trig(phase, sine, true);
        su4q_trig(phase, cosine, false);
        cb_wait_front(sine, 1);
        cb_wait_front(cosine, 1);
        for (uint32_t lane = 0; lane < 8; ++lane) {
            const bool imaginary = (lane & 1U) != 0;
            const uint32_t partner = state + (imaginary ? lane - 1 : lane + 1);
            tile_regs_acquire();
            su4q_first_product(cosine, state + lane);
            su4q_accumulate_product(sine, partner, !imaginary);
            su4q_pack_one(output + lane);
        }
        cb_pop_front(sine, 1);
        cb_pop_front(cosine, 1);
        for (uint32_t cb = phase; cb < state + 8; ++cb) cb_pop_front(cb, 1);
    }
}
