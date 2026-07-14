# Benchmark reports

This directory is the public, evidence-backed benchmark surface for
`tt-rqm-kernels`. Every published number must be traceable to a hashed source
artifact and a versioned release manifest.

## Flagship H1 report

- [Fused Time-Ordered SU(2) Composition on Tenstorrent Wormhole](su2-compose-bench.md)
- Qualification: **Claim Level 1 — qualified first comparison sample**
- Stability: `stable_benchmark=false`
- Public cold-start sessions: **1**
- Family: `SU2HamiltonianBench`; first stage: `SU2ComposeBench`

## Quaternion kernel report

- [Structured FP32 Quaternion Kernels on Tenstorrent Wormhole](wormhole-qmul.md)
- [Wormhole qmul hardware evidence](wormhole-qmul-hardware-evidence.md)
- Qualification: **Claim Level 2 — stable one-device performance**
- Stability: `stable_benchmark=true` for the aggregate release
- Public persistent sessions: **3**

The release records real Wormhole device-0 execution, whole-output correctness,
and three-session stability. It is not a CPU comparison, acceleration claim,
hardware-bandwidth result, energy result, application speedup, dual-device
result, or Tenstorrent endorsement.

## Two-qubit reference foundation

- [EntanglementDynamicsBench reference foundation](entanglement-dynamics-bench.md)
- Family: `TwoQubitHamiltonianBench`; first stage: `EntanglementDynamicsBench`
- Status: CPU reference implemented
- Claim level: none
- Hardware and performance evidence: none
- [Preregistration](../../benchmarks/manifests/entanglement-dynamics-preregistration.json)

## Policy and next measurements

- [Claim policy](claim-policy.md)
- [Preregistered evidence program](methodology.md)
- [Stage B stability thresholds](../stage-b-stability-methodology.md)
- [Wormhole qmul hardware evidence](wormhole-qmul-hardware-evidence.md)
- [Level 2 release manifest](../../benchmarks/manifests/wormhole-qmul-level2.json)
- [Archived Level 1 release manifest](../../benchmarks/manifests/wormhole-qmul.json)
- [SU2ComposeBench release manifest](../../benchmarks/manifests/wormhole-su2-compose.json)
- [SU2 stability methodology](../su2-stability-methodology.md)
- [Frozen SU2 stability preregistration](../../benchmarks/manifests/su2-compose-stability-preregistration.json)

Validate the complete published release without hardware access:

```bash
python scripts/reproduce_wormhole_qmul.py --check
python scripts/reproduce_wormhole_su2_compose.py --check
python scripts/validate_entanglement_dynamics_preregistration.py
```
