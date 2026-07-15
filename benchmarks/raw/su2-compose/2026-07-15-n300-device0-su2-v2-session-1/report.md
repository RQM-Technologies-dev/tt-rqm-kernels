# Fused Time-Ordered SU(2) Composition on Tenstorrent Wormhole

> **RQM runs fused time-ordered SU(2) evolution for two-level Hamiltonian simulation on Tenstorrent Wormhole.**

H1 lowers piecewise-constant two-level Hamiltonian coefficients into FP32 rotors and phase pairs on the CPU. Wormhole performs their ordered composition. H2 will address device-side Hamiltonian coefficient lowering. H1 is a real stage of a Hamiltonian-simulation pipeline, not the complete device-side pipeline.

Stage: `performance`  
Performance eligible: `true`  
Stable benchmark: `false`

Designated SU2ComposeBench stability session; frozen eight-case order, two warmup pairs, ten measured pairs, and no discarded runs.

## Results

| B | K | values/path | fused median s | unfused median s | ratio | fused max error |
|---:|---:|---:|---:|---:|---:|---:|
| 32768 | 8 | 196608 | 0.000140731 | 0.000650966 | 0.216188 | 3.700e-07 |
| 8192 | 32 | 49152 | 0.000420769 | 0.001975398 | 0.213005 | 7.396e-07 |
| 2048 | 128 | 12288 | 0.001577099 | 0.007344533 | 0.214731 | 1.269e-06 |
| 512 | 512 | 3072 | 0.006181396 | 0.026952383 | 0.229345 | 1.868e-06 |
| 1024 | 128 | 6144 | 0.001575278 | 0.007223558 | 0.218075 | 1.028e-06 |
| 4096 | 128 | 24576 | 0.001580660 | 0.007581614 | 0.208486 | 1.436e-06 |
| 16384 | 128 | 98304 | 0.001585800 | 0.007725559 | 0.205267 | 1.455e-06 |
| 65536 | 128 | 393216 | 0.003150771 | 0.014769895 | 0.213324 | 1.644e-06 |

This H1 report does not claim stability, acceleration, CPU superiority, measured bandwidth, or full device-side Hamiltonian lowering.
