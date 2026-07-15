# Fused Time-Ordered SU(2) Composition on Tenstorrent Wormhole

> **RQM runs quantum Hamiltonian simulations on Tenstorrent.**

The first implementation executes fused, time-ordered SU(2) evolution on Wormhole using CPU-lowered FP32 evolution operators. A later stage will lower Hamiltonian coefficients on device.

Stage: `performance`  
Performance eligible: `true`  
Stable benchmark: `false`

New candidate performance experiment on real N300 device 0; candidate SHA-256 54b91bd921a67bcbda0faaafc2019bbfb931a7f1ef5cef26913d252d0f01da16; two warmup pairs and ten measured pairs per case; no stability or acceleration claim.

## Results

| B | K | values/path | fused median s | unfused median s | ratio | fused max error |
|---:|---:|---:|---:|---:|---:|---:|
| 32768 | 8 | 196608 | 0.000140627 | 0.000661848 | 0.212477 | 3.700e-07 |
| 8192 | 32 | 49152 | 0.000418499 | 0.001962468 | 0.213251 | 7.396e-07 |
| 2048 | 128 | 12288 | 0.001575922 | 0.007264874 | 0.216923 | 1.269e-06 |
| 512 | 512 | 3072 | 0.006181302 | 0.026764962 | 0.230948 | 1.868e-06 |
| 1024 | 128 | 6144 | 0.001574468 | 0.007233420 | 0.217666 | 1.028e-06 |
| 4096 | 128 | 24576 | 0.001580886 | 0.007607623 | 0.207803 | 1.436e-06 |
| 16384 | 128 | 98304 | 0.001586488 | 0.007857681 | 0.201903 | 1.455e-06 |
| 65536 | 128 | 393216 | 0.003149216 | 0.015080039 | 0.208833 | 1.644e-06 |

This H1 report does not claim stability, acceleration, CPU superiority, measured bandwidth, or full device-side Hamiltonian lowering.
