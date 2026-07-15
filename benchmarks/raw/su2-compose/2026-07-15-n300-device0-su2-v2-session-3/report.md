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
| 32768 | 8 | 196608 | 0.000126278 | 0.000449378 | 0.281006 | 3.700e-07 |
| 8192 | 32 | 49152 | 0.000408906 | 0.001406018 | 0.290826 | 7.396e-07 |
| 2048 | 128 | 12288 | 0.001562383 | 0.005279469 | 0.295936 | 1.269e-06 |
| 512 | 512 | 3072 | 0.006181147 | 0.027209635 | 0.227168 | 1.868e-06 |
| 1024 | 128 | 6144 | 0.001574096 | 0.007184799 | 0.219087 | 1.028e-06 |
| 4096 | 128 | 24576 | 0.001580611 | 0.007699782 | 0.205280 | 1.436e-06 |
| 16384 | 128 | 98304 | 0.001586113 | 0.007890912 | 0.201005 | 1.455e-06 |
| 65536 | 128 | 393216 | 0.003146161 | 0.015120331 | 0.208075 | 1.644e-06 |

This H1 report does not claim stability, acceleration, CPU superiority, measured bandwidth, or full device-side Hamiltonian lowering.
