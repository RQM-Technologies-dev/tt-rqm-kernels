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
| 32768 | 8 | 196608 | 0.000141318 | 0.000674296 | 0.209578 | 3.700e-07 |
| 8192 | 32 | 49152 | 0.000418666 | 0.001985808 | 0.210829 | 7.396e-07 |
| 2048 | 128 | 12288 | 0.001575780 | 0.007289890 | 0.216160 | 1.269e-06 |
| 512 | 512 | 3072 | 0.006180979 | 0.026882663 | 0.229924 | 1.868e-06 |
| 1024 | 128 | 6144 | 0.001572886 | 0.006748944 | 0.233057 | 1.028e-06 |
| 4096 | 128 | 24576 | 0.001579456 | 0.007624941 | 0.207143 | 1.436e-06 |
| 16384 | 128 | 98304 | 0.001585355 | 0.007832282 | 0.202413 | 1.455e-06 |
| 65536 | 128 | 393216 | 0.003149596 | 0.015367698 | 0.204949 | 1.644e-06 |

This H1 report does not claim stability, acceleration, CPU superiority, measured bandwidth, or full device-side Hamiltonian lowering.
