# Fused Time-Ordered SU(2) Composition on Tenstorrent Wormhole

> **RQM runs fused time-ordered SU(2) evolution for two-level Hamiltonian simulation on Tenstorrent Wormhole.**

H1 lowers piecewise-constant two-level Hamiltonian coefficients into FP32 rotors and phase pairs on the CPU. Wormhole performs their ordered composition. H2 will address device-side Hamiltonian coefficient lowering. H1 is a real stage of a Hamiltonian-simulation pipeline, not the complete device-side pipeline.

Stage: `profile`  
Performance eligible: `true`  
Stable benchmark: `false`

Diagnostic Device Program Profiler and Tracy capture; one fused/unfused pair, no timing warmups, no stability or acceleration claim.

## Results

| B | K | values/path | fused median s | unfused median s | ratio | fused max error |
|---:|---:|---:|---:|---:|---:|---:|
| 32768 | 8 | 196608 | 0.032613195 | 0.034284938 | 0.951240 | 3.700e-07 |

This H1 report does not claim stability, acceleration, CPU superiority, measured bandwidth, or full device-side Hamiltonian lowering.
