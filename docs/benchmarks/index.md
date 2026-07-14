# Benchmark reports

This directory is the public, evidence-backed benchmark surface for
`tt-rqm-kernels`. Every published number must be traceable to a hashed source
artifact and a versioned release manifest.

## Current report

- [Structured FP32 Quaternion Kernels on Tenstorrent Wormhole](wormhole-qmul.md)
- Qualification: **Claim Level 1 — qualified first performance sample**
- Stability: `stable_benchmark=false`
- Public persistent sessions: **1**

The current report records real Wormhole device-0 execution and whole-output
correctness. It is not a stability result, CPU comparison, acceleration claim,
hardware bandwidth result, energy result, application speedup, dual-device
result, or Tenstorrent endorsement.

## SU2HamiltonianBench report

- [Fused Time-Ordered SU(2) Composition on Tenstorrent Wormhole](su2-compose-bench.md)
- Qualification: **Claim Level 1 — qualified first comparison sample**
- Stability: `stable_benchmark=false`
- Public cold-start sessions: **1**
- Family: `SU2HamiltonianBench`; first stage: `SU2ComposeBench`

## Policy and next measurements

- [Claim policy](claim-policy.md)
- [Preregistered evidence program](methodology.md)
- [Stage B stability thresholds](../stage-b-stability-methodology.md)
- [Machine-readable release manifest](../../benchmarks/manifests/wormhole-qmul.json)
- [SU2ComposeBench release manifest](../../benchmarks/manifests/wormhole-su2-compose.json)

Validate the complete published release without hardware access:

```bash
python scripts/reproduce_wormhole_qmul.py --check
python scripts/reproduce_wormhole_su2_compose.py --check
```
