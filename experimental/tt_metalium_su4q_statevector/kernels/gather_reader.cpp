// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

namespace {
uint32_t expand_quartet(uint32_t compact, uint32_t q0, uint32_t q1, uint32_t qubits) {
    uint32_t base = 0;
    uint32_t source_bit = 0;
    for (uint32_t bit = 0; bit < qubits; ++bit) {
        if (bit == q0 || bit == q1) continue;
        base |= ((compact >> source_bit) & 1U) << bit;
        ++source_bit;
    }
    return base;
}
}  // namespace

void kernel_main() {
    const uint32_t state_addr = get_arg_val<uint32_t>(0);
    const uint32_t blocks_addr = get_arg_val<uint32_t>(1);
    const uint32_t tile_count = get_arg_val<uint32_t>(2);
    const uint32_t start_tile = get_arg_val<uint32_t>(3);
    const uint32_t total_quartets = get_arg_val<uint32_t>(4);
    const uint32_t quartets_per_state = get_arg_val<uint32_t>(5);
    const uint32_t amplitudes_per_state = get_arg_val<uint32_t>(6);
    const uint32_t state_component_tiles = get_arg_val<uint32_t>(7);
    const uint32_t block_component_tiles = get_arg_val<uint32_t>(8);
    const uint32_t layer = get_arg_val<uint32_t>(9);
    const uint32_t q0 = get_arg_val<uint32_t>(10);
    const uint32_t q1 = get_arg_val<uint32_t>(11);
    const uint32_t qubits = get_arg_val<uint32_t>(12);
    constexpr auto state_args = TensorAccessorArgs<0>();
    constexpr auto block_args = TensorAccessorArgs<state_args.next_compile_time_args_offset()>();
    const auto state = TensorAccessor(state_args, state_addr);
    const auto blocks = TensorAccessor(block_args, blocks_addr);
    constexpr uint32_t tile_elements = 1024;
    constexpr uint32_t state_cb = 4;
    constexpr uint32_t stage_starts[8] = {11, 15, 8, 9, 10, 0, 4, 19};
    constexpr uint32_t stage_counts[8] = {4, 4, 1, 1, 1, 4, 4, 1};
    for (uint32_t local_tile = 0; local_tile < tile_count; ++local_tile) {
        const uint32_t work_tile = start_tile + local_tile;
        for (uint32_t lane = 0; lane < 8; ++lane) {
            cb_reserve_back(state_cb + lane, 1);
            volatile tt_l1_ptr uint32_t* target =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(state_cb + lane));
            for (uint32_t element = 0; element < tile_elements; ++element) target[element] = 0;
        }
        for (uint32_t element = 0; element < tile_elements; ++element) {
            const uint32_t quartet = work_tile * tile_elements + element;
            if (quartet >= total_quartets) break;
            const uint32_t batch = quartet / quartets_per_state;
            const uint32_t compact = quartet % quartets_per_state;
            const uint32_t base = expand_quartet(compact, q0, q1, qubits);
            const uint32_t amplitudes[4] = {
                base, base | (1U << q0), base | (1U << q1),
                base | (1U << q0) | (1U << q1)};
            for (uint32_t amplitude = 0; amplitude < 4; ++amplitude) {
                const uint32_t flat = batch * amplitudes_per_state + amplitudes[amplitude];
                const uint32_t page_offset = flat % tile_elements;
                const uint32_t page = flat / tile_elements;
                for (uint32_t component = 0; component < 2; ++component) {
                    const uint32_t lane = 2 * amplitude + component;
                    noc_async_read(
                        state.get_noc_addr(
                            component * state_component_tiles + page,
                            page_offset * sizeof(uint32_t)),
                        get_write_ptr(state_cb + lane) + element * sizeof(uint32_t),
                        sizeof(uint32_t));
                }
            }
        }
        noc_async_read_barrier();
        for (uint32_t lane = 0; lane < 8; ++lane) cb_push_back(state_cb + lane, 1);
        for (uint32_t stage = 0; stage < 8; ++stage) {
            for (uint32_t lane = 0; lane < stage_counts[stage]; ++lane) {
                cb_reserve_back(lane, 1);
                noc_async_read_page(
                    (layer * 20 + stage_starts[stage] + lane) * block_component_tiles + work_tile,
                    blocks,
                    get_write_ptr(lane));
            }
            noc_async_read_barrier();
            for (uint32_t lane = 0; lane < stage_counts[stage]; ++lane) cb_push_back(lane, 1);
        }
    }
}
