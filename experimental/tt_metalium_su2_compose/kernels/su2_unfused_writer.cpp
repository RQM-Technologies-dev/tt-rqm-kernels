// SPDX-FileCopyrightText: 2026 RQM Technologies LLC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t tile_count = get_arg_val<uint32_t>(1);
    const uint32_t start_tile = get_arg_val<uint32_t>(2);
    const uint32_t component_tiles = get_arg_val<uint32_t>(3);
    constexpr uint32_t first_output_cb = static_cast<uint32_t>(tt::CBIndex::c_16);
    constexpr auto output_args = TensorAccessorArgs<0>();
    const auto output = TensorAccessor(output_args, output_addr);

    for (uint32_t local_tile = 0; local_tile < tile_count; ++local_tile) {
        const uint32_t tile = start_tile + local_tile;
        for (uint32_t lane = 0; lane < 6; ++lane) {
            cb_wait_front(first_output_cb + lane, 1);
            noc_async_write_page(lane * component_tiles + tile, output, get_read_ptr(first_output_cb + lane));
        }
        noc_async_write_barrier();
        for (uint32_t lane = 0; lane < 6; ++lane) cb_pop_front(first_output_cb + lane, 1);
    }
}
