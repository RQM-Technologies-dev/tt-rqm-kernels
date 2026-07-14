# Fused Time-Ordered SU(2) Composition on Tenstorrent Wormhole

> **RQM runs quantum Hamiltonian simulations on Tenstorrent.**

The first implementation executes fused, time-ordered SU(2) evolution on Wormhole using CPU-lowered FP32 evolution operators. A later stage will lower Hamiltonian coefficients on device.

Stage: `performance`
Performance eligible: `true`
Stable benchmark: `false`

Claim Level 1 first fused/unfused comparison session on N300 device 0; preregistered cases, repeat counts, two warmup pairs, ten measured pairs, alternating order, stable_benchmark=false.

## Results

| B | K | values/path | fused median s | unfused median s | ratio | fused max error |
|---:|---:|---:|---:|---:|---:|---:|
| 32768 | 8 | 196608 | 0.000140680 | 0.000667013 | 0.210911 | 3.700e-07 |
| 8192 | 32 | 49152 | 0.000421271 | 0.001976633 | 0.213126 | 7.396e-07 |
| 2048 | 128 | 12288 | 0.001575422 | 0.007256378 | 0.217109 | 1.269e-06 |
| 512 | 512 | 3072 | 0.006179520 | 0.026700448 | 0.231439 | 1.868e-06 |
| 1024 | 128 | 6144 | 0.001573491 | 0.007091179 | 0.221894 | 1.028e-06 |
| 4096 | 128 | 24576 | 0.001580760 | 0.007650305 | 0.206627 | 1.436e-06 |
| 16384 | 128 | 98304 | 0.001585101 | 0.007769173 | 0.204024 | 1.455e-06 |
| 65536 | 128 | 393216 | 0.003155403 | 0.017643155 | 0.178846 | 1.644e-06 |

This H1 report does not claim stability, acceleration, CPU superiority, measured bandwidth, or full device-side Hamiltonian lowering.
