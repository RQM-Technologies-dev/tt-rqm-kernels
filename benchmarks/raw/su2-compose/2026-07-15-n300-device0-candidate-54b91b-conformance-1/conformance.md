# Fused Time-Ordered SU(2) Composition on Tenstorrent Wormhole

> **RQM runs fused time-ordered SU(2) evolution for two-level Hamiltonian simulation on Tenstorrent Wormhole.**

H1 lowers piecewise-constant two-level Hamiltonian coefficients into FP32 rotors and phase pairs on the CPU. Wormhole performs their ordered composition. H2 will address device-side Hamiltonian coefficient lowering. H1 is a real stage of a Hamiltonian-simulation pipeline, not the complete device-side pipeline.

Stage: `conformance`  
Performance eligible: `true`  
Stable benchmark: `false`

New candidate conformance experiment on real N300 device 0; candidate SHA-256 54b91bd921a67bcbda0faaafc2019bbfb931a7f1ef5cef26913d252d0f01da16; no stability or acceleration claim.

## Results

| B | K | values/path | fused median s | unfused median s | ratio | fused max error |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 8 | 192 | 0.017798219 | 0.016692291 | 1.066254 | 1.417e-07 |
| 2048 | 8 | 12288 | 0.000748012 | 0.001188667 | 0.629286 | 2.969e-07 |

This H1 report does not claim stability, acceleration, CPU superiority, measured bandwidth, or full device-side Hamiltonian lowering.
