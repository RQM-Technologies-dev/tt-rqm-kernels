# Fused Time-Ordered SU(2) Composition on Tenstorrent Wormhole

> **RQM runs fused time-ordered SU(2) evolution for two-level Hamiltonian simulation on Tenstorrent Wormhole.**

H1 lowers piecewise-constant two-level Hamiltonian coefficients into FP32 rotors and phase pairs on the CPU. Wormhole performs their ordered composition. H2 will address device-side Hamiltonian coefficient lowering. H1 is a real stage of a Hamiltonian-simulation pipeline, not the complete device-side pipeline.

Stage: `performance`  
Performance eligible: `true`  
Stable benchmark: `false`

Designated SU2ComposeBench Level 2 v3 fused-only stability session; frozen candidate, source/runtime identity, host contract, five warmups, ten raw-duration samples, isolated runtime cache, and no individual stability claim.

## Results

| B | K | values/path | fused median s | fused max error |
|---:|---:|---:|---:|---:|
| 32768 | 8 | 196608 | 0.000098153 | 3.700e-07 |
| 8192 | 32 | 49152 | 0.000379754 | 7.396e-07 |
| 2048 | 128 | 12288 | 0.001534273 | 1.269e-06 |
| 512 | 512 | 3072 | 0.006150792 | 1.868e-06 |
| 1024 | 128 | 6144 | 0.001534183 | 1.028e-06 |
| 4096 | 128 | 24576 | 0.001534508 | 1.436e-06 |
| 16384 | 128 | 98304 | 0.001536946 | 1.455e-06 |
| 65536 | 128 | 393216 | 0.003081807 | 1.644e-06 |

This H1 report does not claim stability, acceleration, CPU superiority, measured bandwidth, or full device-side Hamiltonian lowering.
