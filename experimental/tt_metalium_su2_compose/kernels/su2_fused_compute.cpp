// SPDX-FileCopyrightText: 2026 RQM Technologies LLC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "su2_compute_common.h"

void kernel_main() {
    const uint32_t tile_count = get_arg_val<uint32_t>(0);
    const uint32_t steps = get_arg_val<uint32_t>(1);
    constexpr uint32_t input = static_cast<uint32_t>(tt::CBIndex::c_0);
    constexpr uint32_t final_output = static_cast<uint32_t>(tt::CBIndex::c_6);
    constexpr uint32_t bank_a = static_cast<uint32_t>(tt::CBIndex::c_16);
    constexpr uint32_t bank_b = static_cast<uint32_t>(tt::CBIndex::c_24);
    init_sfpu(input, bank_a);

    for (uint32_t tile = 0; tile < tile_count; ++tile) {
        for (uint32_t lane = 0; lane < 6; ++lane) cb_wait_front(input + lane, 1);
        for (uint32_t lane = 0; lane < 6; ++lane) {
            tile_regs_acquire();
            copy_tile(input + lane, 0, 0);
            su2_pack_component(bank_a + lane);
        }
        for (uint32_t lane = 0; lane < 6; ++lane) cb_pop_front(input + lane, 1);

        uint32_t accumulator = bank_a;
        for (uint32_t step = 1; step < steps; ++step) {
            const uint32_t output = accumulator == bank_a ? bank_b : bank_a;
            for (uint32_t lane = 0; lane < 6; ++lane) {
                cb_wait_front(input + lane, 1);
                cb_wait_front(accumulator + lane, 1);
            }
            for (uint32_t lane = 0; lane < 6; ++lane) {
                su2_calculate_and_pack(input, accumulator, output, lane);
            }
            for (uint32_t lane = 0; lane < 6; ++lane) {
                cb_pop_front(input + lane, 1);
                cb_pop_front(accumulator + lane, 1);
            }
            accumulator = output;
        }
        for (uint32_t lane = 0; lane < 6; ++lane) cb_wait_front(accumulator + lane, 1);
        for (uint32_t lane = 0; lane < 6; ++lane) {
            tile_regs_acquire();
            copy_tile(accumulator + lane, 0, 0);
            su2_pack_component(final_output + lane);
        }
        for (uint32_t lane = 0; lane < 6; ++lane) cb_pop_front(accumulator + lane, 1);
    }
}
