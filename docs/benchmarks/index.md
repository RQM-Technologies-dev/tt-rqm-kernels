# Benchmark reports

This directory is the public, evidence-backed benchmark surface for
`tt-rqm-kernels`. Every published number must be traceable to a hashed source
artifact and a versioned release manifest.

## Current report

- [Structured FP32 Quaternion Kernels on Tenstorrent Wormhole](wormhole-qmul.md)
- [Wormhole qmul hardware evidence](wormhole-qmul-hardware-evidence.md)
- Qualification: **Claim Level 2 — stable one-device performance**
- Public persistent sessions: **3**

The release records real Wormhole device-0 execution, whole-output correctness,
and three-session stability. It is not a CPU comparison, acceleration claim,
hardware-bandwidth result, energy result, application speedup, dual-device
result, or Tenstorrent endorsement.

## Preregistered next benchmark

- [Fused Time-Ordered SU(2) Composition on Tenstorrent Wormhole](su2-compose-bench.md)
- Current state: CPU contract and preregistration; no SU(2) hardware result yet
- Family: `SU2HamiltonianBench`; first stage: `SU2ComposeBench`

## Policy and next measurements

- [Claim policy](claim-policy.md)
- [Preregistered evidence program](methodology.md)
- [Stage B stability thresholds](../stage-b-stability-methodology.md)
- [Wormhole qmul hardware evidence](wormhole-qmul-hardware-evidence.md)
- [Level 2 release manifest](../../benchmarks/manifests/wormhole-qmul-level2.json)
- [Archived Level 1 release manifest](../../benchmarks/manifests/wormhole-qmul.json)

Validate the complete published release without hardware access:

```bash
python scripts/reproduce_wormhole_qmul.py --check
```
