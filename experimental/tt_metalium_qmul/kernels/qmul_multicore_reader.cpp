// SPDX-FileCopyrightText: 2026 RQM Technologies LLC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_arg_val<uint32_t>(1);
    const uint32_t tile_count = get_arg_val<uint32_t>(2);
    const uint32_t start_tile = get_arg_val<uint32_t>(3);
    const uint32_t component_tiles = get_arg_val<uint32_t>(4);

    constexpr uint32_t first_input_cb = static_cast<uint32_t>(tt::CBIndex::c_0);
    constexpr auto a_args = TensorAccessorArgs<0>();
    const auto a = TensorAccessor(a_args, a_addr);
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto b = TensorAccessor(b_args, b_addr);

    for (uint32_t local_tile = 0; local_tile < tile_count; ++local_tile) {
        const uint32_t tile = start_tile + local_tile;
        for (uint32_t lane = 0; lane < 4; ++lane) {
            const uint32_t page = lane * component_tiles + tile;
            const uint32_t a_cb = first_input_cb + lane;
            const uint32_t b_cb = first_input_cb + 4 + lane;
            cb_reserve_back(a_cb, 1);
            cb_reserve_back(b_cb, 1);
            noc_async_read_page(page, a, get_write_ptr(a_cb));
            noc_async_read_page(page, b, get_write_ptr(b_cb));
        }
        noc_async_read_barrier();
        for (uint32_t lane = 0; lane < 8; ++lane) {
            cb_push_back(first_input_cb + lane, 1);
        }
    }
}
