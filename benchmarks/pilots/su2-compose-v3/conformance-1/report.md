# Fused Time-Ordered SU(2) Composition on Tenstorrent Wormhole

> **RQM runs fused time-ordered SU(2) evolution for two-level Hamiltonian simulation on Tenstorrent Wormhole.**

H1 lowers piecewise-constant two-level Hamiltonian coefficients into FP32 rotors and phase pairs on the CPU. Wormhole performs their ordered composition. H2 will address device-side Hamiltonian coefficient lowering. H1 is a real stage of a Hamiltonian-simulation pipeline, not the complete device-side pipeline.

Stage: `conformance`  
Performance eligible: `true`  
Stable benchmark: `false`

Non-designated SU2ComposeBench Level 2 v3 fused-only conformance; isolated runtime cache and no timing or stability claim.

## Results

| B | K | values/path | fused median s | fused max error |
|---:|---:|---:|---:|---:|
| 32 | 8 | 192 | 2.502557797 | 1.417e-07 |
| 2048 | 8 | 12288 | 0.000728652 | 2.969e-07 |

This H1 report does not claim stability, acceleration, CPU superiority, measured bandwidth, or full device-side Hamiltonian lowering.
