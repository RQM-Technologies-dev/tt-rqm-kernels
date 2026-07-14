// SPDX-FileCopyrightText: 2026 RQM Technologies LLC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t tile_count = get_arg_val<uint32_t>(1);
    const uint32_t start_tile = get_arg_val<uint32_t>(2);
    const uint32_t component_tiles = get_arg_val<uint32_t>(3);
    const uint32_t steps = get_arg_val<uint32_t>(4);
    constexpr uint32_t first_input_cb = static_cast<uint32_t>(tt::CBIndex::c_0);
    constexpr auto input_args = TensorAccessorArgs<0>();
    const auto input = TensorAccessor(input_args, input_addr);

    for (uint32_t local_tile = 0; local_tile < tile_count; ++local_tile) {
        const uint32_t tile = start_tile + local_tile;
        for (uint32_t step = 0; step < steps; ++step) {
            for (uint32_t lane = 0; lane < 6; ++lane) {
                const uint32_t cb = first_input_cb + lane;
                const uint32_t page = (step * 6 + lane) * component_tiles + tile;
                cb_reserve_back(cb, 1);
                noc_async_read_page(page, input, get_write_ptr(cb));
            }
            noc_async_read_barrier();
            for (uint32_t lane = 0; lane < 6; ++lane) cb_push_back(first_input_cb + lane, 1);
        }
    }
}
