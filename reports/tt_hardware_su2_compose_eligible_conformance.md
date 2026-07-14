# Fused Time-Ordered SU(2) Composition on Tenstorrent Wormhole

> **RQM runs quantum Hamiltonian simulations on Tenstorrent.**

The first implementation executes fused, time-ordered SU(2) evolution on Wormhole using CPU-lowered FP32 evolution operators. A later stage will lower Hamiltonian coefficients on device.

Stage: `conformance`
Performance eligible: `true`
Stable benchmark: `false`

Post-audit eligibility rebuild on N300 device 0; whole-output validation of both paths before performance collection.

## Results

| B | K | values/path | fused median s | unfused median s | ratio | fused max error |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 8 | 192 | 0.017583821 | 0.016856028 | 1.043177 | 1.417e-07 |
| 2048 | 8 | 12288 | 0.000827882 | 0.001144568 | 0.723314 | 2.969e-07 |

This H1 report does not claim stability, acceleration, CPU superiority, measured bandwidth, or full device-side Hamiltonian lowering.
