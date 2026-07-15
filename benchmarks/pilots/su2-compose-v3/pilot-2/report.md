# Fused Time-Ordered SU(2) Composition on Tenstorrent Wormhole

> **RQM runs fused time-ordered SU(2) evolution for two-level Hamiltonian simulation on Tenstorrent Wormhole.**

H1 lowers piecewise-constant two-level Hamiltonian coefficients into FP32 rotors and phase pairs on the CPU. Wormhole performs their ordered composition. H2 will address device-side Hamiltonian coefficient lowering. H1 is a real stage of a Hamiltonian-simulation pipeline, not the complete device-side pipeline.

Stage: `performance`  
Performance eligible: `true`  
Stable benchmark: `false`

Non-designated SU2ComposeBench Level 2 v3 pilot; fused_stability only, five warmups, ten raw-duration samples, isolated runtime cache, and no claim.

## Results

| B | K | values/path | fused median s | fused max error |
|---:|---:|---:|---:|---:|
| 32768 | 8 | 196608 | 0.000098143 | 3.700e-07 |
| 8192 | 32 | 49152 | 0.000379811 | 7.396e-07 |
| 2048 | 128 | 12288 | 0.001534400 | 1.269e-06 |
| 512 | 512 | 3072 | 0.006149573 | 1.868e-06 |
| 1024 | 128 | 6144 | 0.001534112 | 1.028e-06 |
| 4096 | 128 | 24576 | 0.001534425 | 1.436e-06 |
| 16384 | 128 | 98304 | 0.001537169 | 1.455e-06 |
| 65536 | 128 | 393216 | 0.003081275 | 1.644e-06 |

This H1 report does not claim stability, acceleration, CPU superiority, measured bandwidth, or full device-side Hamiltonian lowering.
