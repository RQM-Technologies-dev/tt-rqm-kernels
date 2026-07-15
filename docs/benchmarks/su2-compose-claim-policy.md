# SU2ComposeBench Claim Policy

The current public description is:

> **RQM runs fused time-ordered SU(2) evolution for two-level Hamiltonian simulation on Tenstorrent Wormhole.**

It must be followed in the same introductory block by the H1 boundary: the CPU
lowers piecewise-constant two-level Hamiltonian coefficients into FP32 rotors
and phase pairs; Wormhole composes them in time order; H2 will address
device-side coefficient lowering. H1 is a real pipeline stage, not the complete
device-side pipeline.

The shared repository claim ladder applies:

- Level 0: silicon conformance;
- Level 1: qualified first performance sample, non-stable;
- Level 2: stable one-device performance across exactly three designated
  sessions under a frozen stability methodology;
- Level 3: stable matched-scope fused/unfused comparison;
- Level 4: application workload result;
- Level 5: reviewed upstream contribution.

“Full device-side Hamiltonian evolution lowering” is reserved for H2. No level
permits a general quantum-computing, endorsement, or quantum-hardware claim.
Level 2 requires the hash-bound qualifier to recompute every fused, unfused,
and paired-ratio gate. The historical v1 methodology and retained-candidate v2
methodology are distinct campaigns. V2 retained all three designated sessions
but did not pass its frozen variability gates, so it cannot promote the public
Claim Level 1 release. No level authorizes an acceleration claim.

See the [stability methodology](../su2-stability-methodology.md).
