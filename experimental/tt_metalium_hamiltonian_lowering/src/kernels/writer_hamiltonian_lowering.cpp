// SPDX-FileCopyrightText: 2026 RQM Technologies LLC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t tile_count = get_arg_val<uint32_t>(1);
    const uint32_t component_tiles = get_arg_val<uint32_t>(2);
    constexpr auto output_args = TensorAccessorArgs<0>();
    const auto output = TensorAccessor(output_args, output_addr);
    for (uint32_t tile = 0; tile < tile_count; ++tile) {
        for (uint32_t plane = 0; plane < 6; ++plane) {
            const uint32_t cb = 16 + plane;
            cb_wait_front(cb, 1);
            noc_async_write_page(plane * component_tiles + tile, output, get_read_ptr(cb));
        }
        noc_async_write_barrier();
        for (uint32_t plane = 0; plane < 6; ++plane) cb_pop_front(16 + plane, 1);
    }
}
