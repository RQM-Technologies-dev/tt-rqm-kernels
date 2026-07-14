// SPDX-FileCopyrightText: 2026 RQM Technologies LLC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t lhs_addr = get_arg_val<uint32_t>(0);
    const uint32_t rhs_addr = get_arg_val<uint32_t>(1);
    const uint32_t tile_count = get_arg_val<uint32_t>(2);
    const uint32_t start_tile = get_arg_val<uint32_t>(3);
    const uint32_t component_tiles = get_arg_val<uint32_t>(4);
    const uint32_t lhs_page_base = get_arg_val<uint32_t>(5);
    const uint32_t rhs_page_base = get_arg_val<uint32_t>(6);
    constexpr uint32_t first_input_cb = static_cast<uint32_t>(tt::CBIndex::c_0);
    constexpr auto lhs_args = TensorAccessorArgs<0>();
    const auto lhs = TensorAccessor(lhs_args, lhs_addr);
    constexpr auto rhs_args = TensorAccessorArgs<lhs_args.next_compile_time_args_offset()>();
    const auto rhs = TensorAccessor(rhs_args, rhs_addr);

    for (uint32_t local_tile = 0; local_tile < tile_count; ++local_tile) {
        const uint32_t tile = start_tile + local_tile;
        for (uint32_t lane = 0; lane < 6; ++lane) {
            cb_reserve_back(first_input_cb + lane, 1);
            cb_reserve_back(first_input_cb + 6 + lane, 1);
            noc_async_read_page(lhs_page_base + lane * component_tiles + tile, lhs, get_write_ptr(first_input_cb + lane));
            noc_async_read_page(rhs_page_base + lane * component_tiles + tile, rhs, get_write_ptr(first_input_cb + 6 + lane));
        }
        noc_async_read_barrier();
        for (uint32_t lane = 0; lane < 12; ++lane) cb_push_back(first_input_cb + lane, 1);
    }
}
