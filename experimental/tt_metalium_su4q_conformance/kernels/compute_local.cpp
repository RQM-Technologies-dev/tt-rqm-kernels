// SPDX-FileCopyrightText: 2026 RQM Technologies LLC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_common.h"

namespace {

void local_component(uint32_t output_lane, uint32_t qubit) {
    constexpr uint32_t state = 4;
    const uint32_t output_amplitude = output_lane / 2;
    const bool imaginary = (output_lane & 1U) != 0;
    const uint32_t pair_stride = qubit == 0 ? 1 : 2;
    const uint32_t pair_base =
        qubit == 0 ? (output_amplitude / 2) * 2 : output_amplitude % 2;
    const uint32_t first = pair_base;
    const uint32_t second = pair_base + pair_stride;
    const bool upper = output_amplitude == first;
    const uint32_t first_real = state + 2 * first;
    const uint32_t first_imag = first_real + 1;
    const uint32_t second_real = state + 2 * second;
    const uint32_t second_imag = second_real + 1;
    constexpr uint32_t w = 0;
    constexpr uint32_t x = 1;
    constexpr uint32_t y = 2;
    constexpr uint32_t z = 3;

    if (upper && !imaginary) {
        su4q_first_product(w, first_real);
        su4q_accumulate_product(z, first_imag, false);
        su4q_accumulate_product(y, second_real, true);
        su4q_accumulate_product(x, second_imag, false);
    } else if (upper) {
        su4q_first_product(w, first_imag);
        su4q_accumulate_product(z, first_real, true);
        su4q_accumulate_product(y, second_imag, true);
        su4q_accumulate_product(x, second_real, true);
    } else if (!imaginary) {
        su4q_first_product(y, first_real);
        su4q_accumulate_product(x, first_imag, false);
        su4q_accumulate_product(w, second_real, false);
        su4q_accumulate_product(z, second_imag, true);
    } else {
        su4q_first_product(y, first_imag);
        su4q_accumulate_product(x, first_real, true);
        su4q_accumulate_product(w, second_imag, false);
        su4q_accumulate_product(z, second_real, false);
    }
}

}  // namespace

void kernel_main() {
    const uint32_t tile_count = get_arg_val<uint32_t>(0);
    const uint32_t qubit = get_arg_val<uint32_t>(1);
    constexpr uint32_t output = static_cast<uint32_t>(tt::CBIndex::c_16);
    init_sfpu(0, output);
    for (uint32_t tile = 0; tile < tile_count; ++tile) {
        for (uint32_t cb = 0; cb < 12; ++cb) cb_wait_front(cb, 1);
        for (uint32_t lane = 0; lane < 8; ++lane) {
            tile_regs_acquire();
            local_component(lane, qubit);
            su4q_pack_one(output + lane);
        }
        for (uint32_t cb = 0; cb < 12; ++cb) cb_pop_front(cb, 1);
    }
}
