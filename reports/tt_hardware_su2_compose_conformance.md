# Fused Time-Ordered SU(2) Composition on Tenstorrent Wormhole

> **RQM runs quantum Hamiltonian simulations on Tenstorrent.**

The first implementation executes fused, time-ordered SU(2) evolution on Wormhole using CPU-lowered FP32 evolution operators. A later stage will lower Hamiltonian coefficients on device.

Stage: `conformance`
Performance eligible: `false`
Stable benchmark: `false`

Initial H1 N300 device-0 conformance before performance eligibility; both fused and unfused paths validated in one persistent session.

## Results

| B | K | values/path | fused median s | unfused median s | ratio | fused max error |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 8 | 192 | 2.100196957 | 1.814111116 | 1.157700 | 1.417e-07 |
| 2048 | 8 | 12288 | 0.000895911 | 0.001005830 | 0.890718 | 2.969e-07 |

This H1 report does not claim stability, acceleration, CPU superiority, measured bandwidth, or full device-side Hamiltonian lowering.
