// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t state_addr = get_arg_val<uint32_t>(0);
    const uint32_t tile_count = get_arg_val<uint32_t>(1);
    const uint32_t start_tile = get_arg_val<uint32_t>(2);
    const uint32_t total_amplitudes = get_arg_val<uint32_t>(3);
    const uint32_t amplitudes_per_state = get_arg_val<uint32_t>(4);
    const uint32_t state_component_tiles = get_arg_val<uint32_t>(5);
    const uint32_t q0 = get_arg_val<uint32_t>(6);
    const uint32_t q1 = get_arg_val<uint32_t>(7);
    constexpr auto state_args = TensorAccessorArgs<0>();
    const auto state = TensorAccessor(state_args, state_addr);
    constexpr uint32_t tile_elements = 1024;
    constexpr uint32_t output_cb = 24;
    for (uint32_t local_tile = 0; local_tile < tile_count; ++local_tile) {
        const uint32_t work_tile = start_tile + local_tile;
        for (uint32_t lane = 0; lane < 8; ++lane) cb_wait_front(output_cb + lane, 1);
        volatile tt_l1_ptr uint32_t* scratch =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(15));
        for (uint32_t element = 0; element < tile_elements; ++element) {
            const uint32_t flat = work_tile * tile_elements + element;
            if (flat >= total_amplitudes) {
                scratch[element] = 0;
                scratch[tile_elements + element] = 0;
                continue;
            }
            const uint32_t amplitude = flat % amplitudes_per_state;
            const uint32_t row =
                ((amplitude >> q0) & 1U) | (((amplitude >> q1) & 1U) << 1);
            scratch[element] =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(output_cb + 2 * row))[element];
            scratch[tile_elements + element] =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(output_cb + 2 * row + 1))[element];
        }
        noc_async_write_page(work_tile, state, get_write_ptr(15));
        noc_async_write_page(
            state_component_tiles + work_tile, state, get_write_ptr(15) + tile_elements * 4);
        noc_async_write_barrier();
        for (uint32_t lane = 0; lane < 8; ++lane) cb_pop_front(output_cb + lane, 1);
    }
}
