// SPDX-FileCopyrightText: 2026 RQM Technologies LLC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t work_items = get_arg_val<uint32_t>(1);
    const uint32_t component_tiles = get_arg_val<uint32_t>(2);
    constexpr auto input_args = TensorAccessorArgs<0>();
    const auto input = TensorAccessor(input_args, input_addr);
    for (uint32_t work_item = 0; work_item < work_items; ++work_item) {
        const uint32_t step = work_item / component_tiles;
        const uint32_t batch_tile = work_item % component_tiles;
        for (uint32_t lane = 0; lane < 6; ++lane) {
            const uint32_t page = (step * 6 + lane) * component_tiles + batch_tile;
            cb_reserve_back(lane, 1);
            noc_async_read_page(page, input, get_write_ptr(lane));
        }
        noc_async_read_barrier();
        for (uint32_t lane = 0; lane < 6; ++lane) cb_push_back(lane, 1);
    }
}
