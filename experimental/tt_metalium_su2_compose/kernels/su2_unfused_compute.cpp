// SPDX-FileCopyrightText: 2026 RQM Technologies LLC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "su2_compute_common.h"

void kernel_main() {
    const uint32_t tile_count = get_arg_val<uint32_t>(0);
    constexpr uint32_t lhs = static_cast<uint32_t>(tt::CBIndex::c_0);
    constexpr uint32_t rhs = static_cast<uint32_t>(tt::CBIndex::c_6);
    constexpr uint32_t output = static_cast<uint32_t>(tt::CBIndex::c_16);
    init_sfpu(lhs, output);

    for (uint32_t tile = 0; tile < tile_count; ++tile) {
        for (uint32_t lane = 0; lane < 12; ++lane) cb_wait_front(lhs + lane, 1);
        for (uint32_t lane = 0; lane < 6; ++lane) su2_calculate_and_pack(lhs, rhs, output, lane);
        for (uint32_t lane = 0; lane < 12; ++lane) cb_pop_front(lhs + lane, 1);
    }
}
