# tt-rqm-kernels Plan

This plan separates proven capability from evidence still required for broader
claims. The repository is hardware-backed today; cloud access is no longer an
implementation blocker.

## Current proven capabilities

```text
CPU/PyTorch reference operators: complete
TT-Lang simulator qmul: complete, simulator-only
tt-emule TT-Metalium qmul: complete, emulation-only
Stage A scalar RISC-V N300 conformance: complete
Stage B multicore/SFPU qmul first sweep: Claim Level 1
Persistent-device qmul release: Claim Level 2
Independent qmul multi-session stability: complete
SU2ComposeBench H1 foundation: complete
SU2ComposeBench H1 Wormhole implementation: present
SU2ComposeBench first fused/unfused session: Claim Level 1
Independent SU2 multi-session stability: not established; v2 campaign did not qualify
EntanglementDynamicsBench CPU reference foundation: complete
EntanglementDynamicsBench hardware implementation: not started
qmul profiler and ceiling diagnostics: present
SU2 profiler attribution: complete for retained 54b91b candidate
CPU timing-scope-compatible comparison: pending
Energy measurement: pending
H2 device-side coefficient lowering: pending
qmul upstream placement: pending Tenstorrent guidance in tt-metal#49887
SU2ComposeBench TT-NN integration: deferred until H1 stability
Broader RQM TT-NN operator family: deferred
TT-MLIR lowering discussion: deferred until the backend evidence is sufficiently mature
```

The primary real-device releases use one Wormhole device (`device_id=0`) and
pinned TT-Metalium provenance. Stage A preserves the scalar correctness
baseline. Stage B moves qmul arithmetic into multicore Tensix compute/SFPU
kernels. Three independent persistent qmul sessions pass the preregistered
Level 2 stability gates. H1 adds fused and unfused ordered SU(2)
rotor-and-phase composition but remains at Level 1.

H1 begins after CPU lowering. The CPU converts piecewise-constant two-level
Hamiltonian coefficients into FP32 rotors and phase pairs; Wormhole composes
those operators in exact time order. This is a real stage of the simulation
pipeline, but not full device-side Hamiltonian lowering.

The sibling `TwoQubitHamiltonianBench` now defines a CPU-only
`EntanglementDynamicsBench` reference: Pauli-product Hamiltonian lowering,
ordered joint-state evolution, an independent complex128 oracle, local U(2)
application, and entanglement diagnostics. It has no hardware evidence or claim
level. The bridge is `qmul -> local SU(2) -> U_A tensor U_B -> joint-state
evolution -> nonlocal Hamiltonians -> entanglement metrics`.

## Active evidence-completion work

### SU2ComposeBench stability

The frozen v2 campaign collected all three independent cold-start sessions
with the same candidate, environment, input contract, paired ordering, and
complete correctness. No designated result was discarded or replaced. The
deterministic qualifier rejected five cases under the preregistered fused,
unfused, and paired-ratio variability gates, so Claim Level 2 remains
unavailable and the public release remains Level 1.

A separate real-N300 candidate experiment is retained under
`benchmarks/raw/su2-compose/2026-07-15-n300-device0-candidate-54b91b-*`.
It validates candidate `54b91b…` for two conformance cases and one complete
eight-case paired performance run, but it is not a designated session in the
existing hash-bound campaign. Its packages are now protected by a dedicated
fail-closed candidate-experiment manifest and validator. Profiler attribution
in [issue #27](https://github.com/RQM-Technologies-dev/tt-rqm-kernels/issues/27)
retained the exact candidate because reader, compute, and writer scopes overlap
and no isolated architectural correction was supported. The v2 contract was
frozen before designated session 1. The three retained packages and
[failed qualification](benchmarks/processed/wormhole-su2-compose-stability-qualification.json)
complete [issue #28](https://github.com/RQM-Technologies-dev/tt-rqm-kernels/issues/28)
without promoting the release. Do not combine campaigns or replace a
designated result.

### Profiler attribution

Device Program Profiler and Tracy evidence covers four representative fused and
unfused cases. Reader, compute, and writer scopes overlap in every fused
dispatch; writer is marginally longest but less than five percent beyond the
next-longest role. Circular-buffer wait and SFPU-utilization counters remain
unobservable in the pinned tools. The exact `54b91b…` candidate is retained;
logical-byte formulas remain distinct from measured hardware bandwidth.

### Matched comparisons and energy

A future CPU comparison must use identical serialized FP32 inputs, operation
semantics, validation, batching, warmups, threading disclosure, and timing
boundaries. Energy measurement requires a separate preregistered acquisition
protocol. Neither result exists today.

## Deferred integration work

### qmul upstream placement

The qmul-only placement question is pending Tenstorrent guidance in
[tenstorrent/tt-metal#49887](https://github.com/tenstorrent/tt-metal/issues/49887).
No upstream implementation begins until maintainers choose between a Metalium
example and an experimental TT-NN operation.

### SU2ComposeBench TT-NN integration

SU2ComposeBench TT-NN integration is deferred until the lower-level H1
implementation and evidence are stable. A wrapper must preserve ordered
composition, whole-output validation, provenance, and timing boundaries rather
than hiding them behind a higher-level API.

### Broader RQM TT-NN operator family

A broader RQM operator family remains deferred. It must not inherit qmul or SU2
evidence and is not part of the current upstream placement request.

### TT-MLIR

TT-MLIR lowering discussion is deferred until the backend evidence is mature
enough to identify a small, concrete lowering requirement. The current project
does not request native quaternion hardware or a new datatype.

### Supporting material

TT-Lang, tt-emule, console, cloud-access, delegated-validation, and outreach
documents remain useful setup and historical references. They are indexed in
[docs/index.md](docs/index.md), but they no longer define the active blocker.

## Future technical milestones

### H2: device-side coefficient lowering

H2 will accept Hamiltonian coefficients `[B,K,4]` and `dt` directly, then
perform norm, axis, sine, cosine, phase, rotor construction, and ordered
composition on device. Work begins only after H1 stability and an audit of the
pinned SFPU sine, cosine, reciprocal, square-root, and norm support.

Only H2 may support the phrase “full device-side two-level Hamiltonian evolution
lowering.” This task identifies H2 as the next technical milestone; it does not
implement it.

### Two-qubit hardware work

Any EntanglementDynamicsBench device path requires a separate contract and
preregistration. The present foundation defines no throughput metric,
performance cases, release manifest, or TT-Metalium implementation. Local U(2)
operations preserve entanglement; nonlocal interaction terms can generate it.
`rqm-entanglement` is not a runtime dependency.

### Later benchmark families

`RigidBodyHamiltonianBench`, PoseStreamBench, and broader physical-AI studies
remain separate follow-ons. They must not inherit H1 evidence or claim levels.

## Claim boundaries and non-goals

Individual qmul and SU2 hardware reports remain `stable_benchmark=false`.
The hash-bound aggregate qmul release is `stable_benchmark=true` because three
independent sessions passed its preregistered Level 2 gates. SU2 remains Level
1. Neither state is an acceleration claim.

The repository currently makes no claim of:

- CPU acceleration or superiority;
- stable SU2 one-device performance;
- measured DRAM, NoC, PCIe, or compute bandwidth;
- energy efficiency or application speedup;
- full device-side Hamiltonian coefficient lowering;
- dual-device or aggregate N300 performance;
- arbitrary quantum-circuit simulation or quantum-hardware replacement;
- entanglement execution on Tenstorrent hardware;
- Tenstorrent endorsement.

## Primary references

- [SU2ComposeBench report](docs/benchmarks/su2-compose-bench.md)
- [Wormhole qmul report](docs/benchmarks/wormhole-qmul.md)
- [Hamiltonian roadmap](docs/hamiltonian-evolution-roadmap.md)
- [EntanglementDynamicsBench foundation](docs/benchmarks/entanglement-dynamics-bench.md)
- [Benchmark claim policy](docs/benchmarks/claim-policy.md)
- [Documentation index](docs/index.md)
