# Fused Time-Ordered SU(2) Composition on Tenstorrent Wormhole

> **RQM runs fused time-ordered SU(2) evolution for two-level Hamiltonian simulation on Tenstorrent Wormhole.**

H1 lowers piecewise-constant two-level Hamiltonian coefficients into FP32 rotors and phase pairs on the CPU. Wormhole performs their ordered composition. H2 will address device-side Hamiltonian coefficient lowering. H1 is a real stage of a Hamiltonian-simulation pipeline, not the complete device-side pipeline.

Stage: `performance`  
Performance eligible: `true`  
Stable benchmark: `false`

New candidate performance experiment on real N300 device 0; candidate SHA-256 54b91bd921a67bcbda0faaafc2019bbfb931a7f1ef5cef26913d252d0f01da16; two warmup pairs and ten measured pairs per case; no stability or acceleration claim.

## Results

| B | K | values/path | fused median s | unfused median s | ratio | fused max error |
|---:|---:|---:|---:|---:|---:|---:|
| 32768 | 8 | 196608 | 0.000140694 | 0.000674416 | 0.208617 | 3.700e-07 |
| 8192 | 32 | 49152 | 0.000416085 | 0.001669157 | 0.249279 | 7.396e-07 |
| 2048 | 128 | 12288 | 0.001575804 | 0.007338914 | 0.214719 | 1.269e-06 |
| 512 | 512 | 3072 | 0.006180831 | 0.026630705 | 0.232094 | 1.868e-06 |
| 1024 | 128 | 6144 | 0.001575025 | 0.007308689 | 0.215500 | 1.028e-06 |
| 4096 | 128 | 24576 | 0.001579727 | 0.007642589 | 0.206700 | 1.436e-06 |
| 16384 | 128 | 98304 | 0.001583986 | 0.007775787 | 0.203707 | 1.455e-06 |
| 65536 | 128 | 393216 | 0.003150506 | 0.017847809 | 0.176521 | 1.644e-06 |

This H1 report does not claim stability, acceleration, CPU superiority, measured bandwidth, or full device-side Hamiltonian lowering.
