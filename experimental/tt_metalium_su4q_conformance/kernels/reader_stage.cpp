// SPDX-FileCopyrightText: 2026 RQM Technologies LLC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t block_addr = get_arg_val<uint32_t>(0);
    const uint32_t state_addr = get_arg_val<uint32_t>(1);
    const uint32_t tile_count = get_arg_val<uint32_t>(2);
    const uint32_t start_tile = get_arg_val<uint32_t>(3);
    const uint32_t component_tiles = get_arg_val<uint32_t>(4);
    const uint32_t block_lane_start = get_arg_val<uint32_t>(5);
    const uint32_t parameter_lanes = get_arg_val<uint32_t>(6);
    const uint32_t block_base_page = get_arg_val<uint32_t>(7);
    constexpr auto block_args = TensorAccessorArgs<0>();
    constexpr auto state_args = TensorAccessorArgs<block_args.next_compile_time_args_offset()>();
    const auto block = TensorAccessor(block_args, block_addr);
    const auto state = TensorAccessor(state_args, state_addr);

    for (uint32_t local_tile = 0; local_tile < tile_count; ++local_tile) {
        const uint32_t tile = start_tile + local_tile;
        for (uint32_t lane = 0; lane < parameter_lanes; ++lane) {
            cb_reserve_back(lane, 1);
            noc_async_read_page(
                block_base_page + (block_lane_start + lane) * component_tiles + tile,
                block,
                get_write_ptr(lane));
        }
        for (uint32_t lane = 0; lane < 8; ++lane) {
            const uint32_t cb = parameter_lanes + lane;
            cb_reserve_back(cb, 1);
            noc_async_read_page(lane * component_tiles + tile, state, get_write_ptr(cb));
        }
        noc_async_read_barrier();
        for (uint32_t lane = 0; lane < parameter_lanes + 8; ++lane) {
            cb_push_back(lane, 1);
        }
    }
}
