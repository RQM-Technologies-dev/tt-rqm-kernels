// SPDX-FileCopyrightText: 2026 RQM Technologies LLC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t work_items = get_arg_val<uint32_t>(1);
    const uint32_t component_tiles = get_arg_val<uint32_t>(2);
    constexpr auto output_args = TensorAccessorArgs<0>();
    const auto output = TensorAccessor(output_args, output_addr);
    for (uint32_t work_item = 0; work_item < work_items; ++work_item) {
        const uint32_t step = work_item / component_tiles;
        const uint32_t batch_tile = work_item % component_tiles;
        for (uint32_t lane = 0; lane < 6; ++lane) {
            const uint32_t cb = 16 + lane;
            const uint32_t page = (step * 6 + lane) * component_tiles + batch_tile;
            cb_wait_front(cb, 1);
            noc_async_write_page(page, output, get_read_ptr(cb));
        }
        noc_async_write_barrier();
        for (uint32_t lane = 0; lane < 6; ++lane) cb_pop_front(16 + lane, 1);
    }
}
