# SU2ComposeBench Claim Policy

The short public description is intentionally accessible, while the evidence
state determines its tense.

- Before silicon conformance: **RQM is building quantum Hamiltonian simulation
  benchmarks for Tenstorrent.**
- After Level 0 silicon conformance: **RQM runs quantum Hamiltonian simulations
  on Tenstorrent.**

The second form must be followed in the same introductory block by this scope:

> The first implementation executes fused, time-ordered SU(2) evolution on
> Wormhole using CPU-lowered FP32 evolution operators. A later stage will lower
> Hamiltonian coefficients on device.

The shared repository claim ladder applies:

- Level 0: silicon conformance;
- Level 1: qualified first performance sample, non-stable;
- Level 2: stable one-device performance across at least three sessions;
- Level 3: stable matched-scope fused/unfused comparison;
- Level 4: application workload result;
- Level 5: reviewed upstream contribution.

“Full device-side Hamiltonian evolution lowering” is reserved for H2. No level
permits a general quantum-computing, endorsement, or quantum-hardware claim.
