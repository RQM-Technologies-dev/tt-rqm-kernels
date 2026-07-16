// SPDX-FileCopyrightText: 2026 RQM Technologies LLC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t tile_count = get_arg_val<uint32_t>(1);
    const uint32_t component_tiles = get_arg_val<uint32_t>(2);
    constexpr auto input_args = TensorAccessorArgs<0>();
    const auto input = TensorAccessor(input_args, input_addr);
    for (uint32_t tile = 0; tile < tile_count; ++tile) {
        for (uint32_t plane = 0; plane < 6; ++plane) {
            cb_reserve_back(plane, 1);
            noc_async_read_page(plane * component_tiles + tile, input, get_write_ptr(plane));
        }
        noc_async_read_barrier();
        for (uint32_t plane = 0; plane < 6; ++plane) cb_push_back(plane, 1);
    }
}
