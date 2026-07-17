# Benchmark reports

This directory is the public, evidence-backed benchmark surface for
`tt-rqm-kernels`. Every published number must be traceable to a hashed source
artifact and a versioned release manifest.

## Flagship H1 report

- [Fused Time-Ordered SU(2) Composition on Tenstorrent Wormhole](su2-compose-bench.md)
- Qualification: **Claim Level 2 — stable one-device fused performance**
- Stability: `stable_benchmark=true` for the aggregate release
- Public cold-start sessions: **3**
- Family: `SU2HamiltonianBench`; first stage: `SU2ComposeBench`

## Separate candidate experiment

- [N300 candidate experiment: `54b91b…`](su2-compose-candidate-54b91b.md)
- Status: real hardware conformance plus one eight-case paired performance
  experiment; `stable_benchmark=false`
- This is not part of the hash-bound Level 2 release.

## Current and retained SU2 stability campaigns

- [V3 preregistration](../../benchmarks/manifests/su2-compose-stability-preregistration-v3.json)
- [V3 deterministic qualification](../../benchmarks/processed/wormhole-su2-compose-v3-stability-qualification.json)
- [Level 2 release manifest](../../benchmarks/manifests/wormhole-su2-compose-level2.json)
- Outcome: three designated N300 sessions passed all fused stability gates;
  aggregate `stable_benchmark=true`, source sessions `false`.
- The [retained v2 campaign](../../benchmarks/processed/wormhole-su2-compose-stability-qualification.json)
  remains a historical non-qualifying record.

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

## H2A Claim Level 0 silicon conformance

- Family: `HamiltonianLoweringBench`; stage: H2A coefficient lowering
- Qualification: **Claim Level 0 — silicon conformance**
- Stability and performance eligibility: `stable_benchmark=false`,
  `performance_eligible=false`
- Public designated sessions: **1** on N300 device 0
- [Public H2A conformance report](hamiltonian-lowering-h2a.md)
- [Pre-hardware Claim Level 0 preregistration](../../benchmarks/manifests/hamiltonian-lowering-h2a-preregistration.json)
- [Retained N300 development blocker](../../benchmarks/pilots/hamiltonian-lowering-h2a/h2a-n300-development-blocker-20260716/README.md)
- [Compensated development comparison](../../benchmarks/pilots/hamiltonian-lowering-h2a/h2a-compensated-development-20260716/README.md)
- [Passing non-designated nine-case pilot](../../benchmarks/pilots/hamiltonian-lowering-h2a/h2a-compensated-n300-pilot-20260716/suite-report.md)
- [Clean-build and clean-tree reproduction](../../benchmarks/pilots/hamiltonian-lowering-h2a/h2a-clean-reproduction-20260716/README.md)
- [Frozen designated Claim Level 0 contract](../../benchmarks/manifests/hamiltonian-lowering-h2a-designated-conformance.json)
- [Claim Level 0 release manifest](../../benchmarks/manifests/wormhole-hamiltonian-lowering.json)
- [Deterministic qualification](../../benchmarks/processed/wormhole-hamiltonian-lowering-h2a-qualification.json)

The original large-angle blocker and the later non-designated pilot remain
preserved development evidence. After the contract was frozen, one designated
N300 device-0 session passed all nine cases with one attempt per case and no
retry or replacement. This establishes silicon conformance only. It makes no
performance, stability, full-H2, speedup, bandwidth, energy, dual-device, or
inherited-H1 claim.

## H2B device-resident evolution pilot

- Family: `HamiltonianEvolutionBench`; stage: H2B complete two-level evolution
- Status: Contract-v1 Session 2 retained; did not pass (`runtime`); no H2B
  hardware claim exists
- Architecture: two programs, one device session, device-DRAM intermediate,
  zero intermediate host round trips
- Stability and performance eligibility: `stable_benchmark=false`,
  `performance_eligible=false`, `claim_level=null`
- [H2B foundation report](hamiltonian-evolution-h2b.md)
- [Processed pilot qualification](../../benchmarks/processed/hamiltonian-evolution-h2b-pilot-qualification.json)
- [Pilot blocker](../../reports/h2b_n300_pilot_blocker.md)
- [Session 2 qualification](../../benchmarks/processed/hamiltonian-evolution-h2b-pilot-session-2-qualification.json)
- [Session 2 blocker](../../reports/h2b_n300_pilot_session_2_blocker.md)

H2B does not inherit the H1 Level 2 or H2A Level 0 results. It is not a
single fused kernel or an acceleration result.

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
- [Frozen retained-candidate SU2 stability preregistration](../../benchmarks/manifests/su2-compose-stability-preregistration-v2.json)

Validate the complete published release without hardware access:

```bash
python scripts/reproduce_wormhole_qmul.py --check
python scripts/reproduce_wormhole_su2_compose.py --check
python scripts/validate_entanglement_dynamics_preregistration.py
python scripts/validate_hamiltonian_lowering_preregistration.py
python scripts/validate_hamiltonian_evolution_candidate.py \
  --command "python scripts/hamiltonian_evolution_external_reference.py"
python scripts/validate_repository_claims.py
```
