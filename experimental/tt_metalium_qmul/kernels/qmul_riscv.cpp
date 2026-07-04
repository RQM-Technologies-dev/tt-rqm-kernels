// SPDX-FileCopyrightText: 2026 RQM Technologies LLC
//
// SPDX-License-Identifier: Apache-2.0

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

        float* av = reinterpret_cast<float*>(a_l1);
        float* bv = reinterpret_cast<float*>(b_l1);
        float* ov = reinterpret_cast<float*>(out_l1);

        const float ar = av[0];
        const float ai = av[1];
        const float aj = av[2];
        const float ak = av[3];
        const float br = bv[0];
        const float bi = bv[1];
        const float bj = bv[2];
        const float bk = bv[3];

        ov[0] = ar * br - ai * bi - aj * bj - ak * bk;
        ov[1] = ar * bi + ai * br + aj * bk - ak * bj;
        ov[2] = ar * bj - ai * bk + aj * br + ak * bi;
        ov[3] = ar * bk + ai * bj - aj * bi + ak * br;

        noc_async_write(out_l1, out.get_noc_addr(index), qbytes);
        noc_async_write_barrier();
    }
}
