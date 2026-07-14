# Wormhole qmul profiler and same-device ceilings

Diagnostic evidence. The qmul logical 48-byte model is not reinterpreted as measured DRAM, NoC, or PCIe traffic.

## Profiler

Device Program Profiler and Tracy captured N=65,536 and N=262,144. Reader, compute, and writer scopes overlap on all 56 cores. The writer/NCRISC maximum is marginally longest, with compute nearly coextensive. The pinned tools expose no circular-buffer stall, NoC-wait, or SFPU-utilization counters.

| N | reader max cycles | compute-math max cycles | writer max cycles |
|---:|---:|---:|---:|
| 65,536 | 14560 | 29847 | 30217 |
| 262,144 | 85691 | 100956 | 101328 |

## Pinned microbenchmarks

| measurement | value | status |
|---|---:|---|
| pcie_h2d | 12.129 GB/s | measured |
| pcie_d2h | 9.092 GB/s | measured |
| noc_read | 3.767 B/clock | measured-host-timed |
| noc_write | 23.234 B/clock | measured-host-timed |
| dram_adjacent_read | 124.512 GB/s | measured-host-timed-post-first-use |
| dispatch_all_cores | 15.332 us/iteration | measured |
| compute_bfp8_fpu | 27.060 TFLOP/s | closest-supported-noncomparable |
| compute_fp32_sfpu | not measured | not available in pinned microbenchmark suite |

The BFP8 FPU matmul is the closest supported compute benchmark, not an FP32 SFPU ceiling and not a qmul comparison. Failed pinned benchmark attempts remain in the raw evidence directory.
