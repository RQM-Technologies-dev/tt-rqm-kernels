// SPDX-FileCopyrightText: 2026 RQM Technologies LLC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

namespace {

union FloatWord {
    uint32_t word;
    float value;
};

float word_to_float(uint32_t word) {
    FloatWord converted;
    converted.word = word;
    return converted.value;
}

uint32_t float_to_word(float value) {
    FloatWord converted;
    converted.value = value;
    return converted.word;
}

}  // namespace

void kernel_main() {
    uint32_t a_dram = get_arg_val<uint32_t>(0);
    uint32_t b_dram = get_arg_val<uint32_t>(1);
    uint32_t out_dram = get_arg_val<uint32_t>(2);
    uint32_t a_l1 = get_arg_val<uint32_t>(3);
    uint32_t b_l1 = get_arg_val<uint32_t>(4);
    uint32_t out_l1 = get_arg_val<uint32_t>(5);
    uint32_t items = get_arg_val<uint32_t>(6);

    constexpr uint32_t qbytes = 4 * sizeof(uint32_t);
    InterleavedAddrGen<true> a = {.bank_base_address = a_dram, .page_size = qbytes};
    InterleavedAddrGen<true> b = {.bank_base_address = b_dram, .page_size = qbytes};
    InterleavedAddrGen<true> out = {.bank_base_address = out_dram, .page_size = qbytes};

    for (uint32_t index = 0; index < items; ++index) {
        noc_async_read(a.get_noc_addr(index), a_l1, qbytes);
        noc_async_read(b.get_noc_addr(index), b_l1, qbytes);
        noc_async_read_barrier();

        volatile tt_l1_ptr uint32_t* av = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(3));
        volatile tt_l1_ptr uint32_t* bv = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(4));
        volatile tt_l1_ptr uint32_t* ov = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(5));

        const float ar = word_to_float(av[0]);
        const float ai = word_to_float(av[1]);
        const float aj = word_to_float(av[2]);
        const float ak = word_to_float(av[3]);
        const float br = word_to_float(bv[0]);
        const float bi = word_to_float(bv[1]);
        const float bj = word_to_float(bv[2]);
        const float bk = word_to_float(bv[3]);

        ov[0] = float_to_word(ar * br - ai * bi - aj * bj - ak * bk);
        ov[1] = float_to_word(ar * bi + ai * br + aj * bk - ak * bj);
        ov[2] = float_to_word(ar * bj - ai * bk + aj * br + ak * bi);
        ov[3] = float_to_word(ar * bk + ai * bj - aj * bi + ak * br);

        noc_async_write(out_l1, out.get_noc_addr(index), qbytes);
        noc_async_write_barrier();
    }
}
