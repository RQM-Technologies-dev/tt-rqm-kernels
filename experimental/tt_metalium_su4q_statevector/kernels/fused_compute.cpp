// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "../../tt_metalium_su4q_conformance/kernels/compute_common.h"

namespace {

void local_component(uint32_t param, uint32_t state, uint32_t lane, uint32_t qubit) {
    const uint32_t amplitude = lane / 2;
    const bool imaginary = (lane & 1U) != 0;
    const uint32_t stride = qubit == 0 ? 1 : 2;
    const uint32_t base = qubit == 0 ? (amplitude / 2) * 2 : amplitude % 2;
    const uint32_t first = base;
    const uint32_t second = base + stride;
    const bool upper = amplitude == first;
    const uint32_t fr = state + 2 * first;
    const uint32_t fi = fr + 1;
    const uint32_t sr = state + 2 * second;
    const uint32_t si = sr + 1;
    const uint32_t w = param;
    const uint32_t x = param + 1;
    const uint32_t y = param + 2;
    const uint32_t z = param + 3;
    if (upper && !imaginary) {
        su4q_first_product(w, fr); su4q_accumulate_product(z, fi, false);
        su4q_accumulate_product(y, sr, true); su4q_accumulate_product(x, si, false);
    } else if (upper) {
        su4q_first_product(w, fi); su4q_accumulate_product(z, fr, true);
        su4q_accumulate_product(y, si, true); su4q_accumulate_product(x, sr, true);
    } else if (!imaginary) {
        su4q_first_product(y, fr); su4q_accumulate_product(x, fi, false);
        su4q_accumulate_product(w, sr, false); su4q_accumulate_product(z, si, true);
    } else {
        su4q_first_product(y, fi); su4q_accumulate_product(x, fr, true);
        su4q_accumulate_product(w, si, false); su4q_accumulate_product(z, sr, false);
    }
}

void apply_local(uint32_t state, uint32_t output, uint32_t qubit) {
    for (uint32_t cb = 0; cb < 4; ++cb) cb_wait_front(cb, 1);
    for (uint32_t cb = state; cb < state + 8; ++cb) cb_wait_front(cb, 1);
    for (uint32_t lane = 0; lane < 8; ++lane) {
        tile_regs_acquire();
        local_component(0, state, lane, qubit);
        su4q_pack_one(output + lane);
    }
    for (uint32_t cb = 0; cb < 4; ++cb) cb_pop_front(cb, 1);
    for (uint32_t cb = state; cb < state + 8; ++cb) cb_pop_front(cb, 1);
}

void apply_cartan(uint32_t state, uint32_t output, uint32_t pauli) {
    constexpr uint32_t angle = 0;
    constexpr uint32_t sine = 12;
    constexpr uint32_t cosine = 13;
    cb_wait_front(angle, 1);
    for (uint32_t cb = state; cb < state + 8; ++cb) cb_wait_front(cb, 1);
    su4q_trig(angle, sine, true);
    su4q_trig(angle, cosine, false);
    cb_wait_front(sine, 1); cb_wait_front(cosine, 1);
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
        const uint32_t partner_cb = state + 2 * partner + (imaginary ? 0 : 1);
        tile_regs_acquire();
        su4q_first_product(cosine, state + lane);
        su4q_accumulate_product(sine, partner_cb, imaginary ? !positive : positive);
        su4q_pack_one(output + lane);
    }
    cb_pop_front(sine, 1); cb_pop_front(cosine, 1); cb_pop_front(angle, 1);
    for (uint32_t cb = state; cb < state + 8; ++cb) cb_pop_front(cb, 1);
}

void apply_phase(uint32_t state) {
    constexpr uint32_t phase = 0;
    constexpr uint32_t sine = 12;
    constexpr uint32_t cosine = 13;
    constexpr uint32_t output = 24;
    cb_wait_front(phase, 1);
    for (uint32_t cb = state; cb < state + 8; ++cb) cb_wait_front(cb, 1);
    su4q_trig(phase, sine, true);
    su4q_trig(phase, cosine, false);
    cb_wait_front(sine, 1); cb_wait_front(cosine, 1);
    for (uint32_t lane = 0; lane < 8; ++lane) {
        const bool imaginary = (lane & 1U) != 0;
        const uint32_t partner = state + (imaginary ? lane - 1 : lane + 1);
        tile_regs_acquire();
        su4q_first_product(cosine, state + lane);
        su4q_accumulate_product(sine, partner, !imaginary);
        su4q_pack_one(output + lane);
    }
    cb_pop_front(sine, 1); cb_pop_front(cosine, 1); cb_pop_front(phase, 1);
    for (uint32_t cb = state; cb < state + 8; ++cb) cb_pop_front(cb, 1);
}

}  // namespace

void kernel_main() {
    const uint32_t tile_count = get_arg_val<uint32_t>(0);
    constexpr uint32_t bank_a = 4;
    constexpr uint32_t bank_b = 16;
    init_sfpu(0, bank_b);
    for (uint32_t tile = 0; tile < tile_count; ++tile) {
        cb_wait_front(14, 32);
        cb_pop_front(14, 32);
        cb_wait_front(14, 32);
        cb_pop_front(14, 32);
        apply_local(bank_a, bank_b, 0);
        apply_local(bank_b, bank_a, 1);
        apply_cartan(bank_a, bank_b, 0);
        apply_cartan(bank_b, bank_a, 1);
        apply_cartan(bank_a, bank_b, 2);
        apply_local(bank_b, bank_a, 0);
        apply_local(bank_a, bank_b, 1);
        apply_phase(bank_b);
    }
}
