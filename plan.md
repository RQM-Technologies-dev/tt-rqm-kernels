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
Stage B multicore/SFPU qmul: Claim Level 1
Persistent-device qmul session: Claim Level 1
SU2ComposeBench H1 foundation: complete
SU2ComposeBench H1 Wormhole implementation: present
SU2ComposeBench first fused/unfused session: Claim Level 1
Independent multi-session stability: pending
Profiler attribution: pending
CPU timing-scope-compatible comparison: pending
Energy measurement: pending
H2 device-side coefficient lowering: pending
TT-NN integration: deferred
TT-MLIR lowering discussion: deferred until the backend evidence is sufficiently mature
```

The real-device evidence uses one Wormhole device (`device_id=0`) and pinned
TT-Metalium provenance. Stage A preserves the scalar correctness baseline.
Stage B moves qmul arithmetic into multicore Tensix compute/SFPU kernels. H1
adds fused and unfused ordered SU(2) rotor-and-phase composition.

H1 begins after CPU lowering. The CPU converts piecewise-constant two-level
Hamiltonian coefficients into FP32 rotors and phase pairs; Wormhole composes
those operators in exact time order. This is a real stage of the simulation
pipeline, but not full device-side Hamiltonian lowering.

## Active evidence-completion work

### SU2ComposeBench stability

Collect three independent cold-start sessions with the same candidate,
environment, input contract, paired ordering, and complete correctness. Do not
discard failed or noisy designated sessions. Claim Level 2 remains unavailable
until the preregistered coefficient-of-variation gates pass.

### Profiler attribution

Capture Device Program Profiler and Tracy evidence for the fused and unfused
paths. Attribute dispatch, compute, data movement, and synchronization costs
before selecting additional optimizations. Logical-byte formulas must remain
distinct from measured hardware bandwidth.

### Matched comparisons and energy

A future CPU comparison must use identical serialized FP32 inputs, operation
semantics, validation, batching, warmups, threading disclosure, and timing
boundaries. Energy measurement requires a separate preregistered acquisition
protocol. Neither result exists today.

## Deferred integration work

### TT-NN

TT-NN integration is deferred until the lower-level H1 implementation and
evidence are stable. A wrapper must preserve ordered composition, whole-output
validation, provenance, and timing boundaries rather than hiding them behind a
higher-level API.

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

### Later benchmark families

`RigidBodyHamiltonianBench`, PoseStreamBench, and broader physical-AI studies
remain separate follow-ons. They must not inherit H1 evidence or claim levels.

## Claim boundaries and non-goals

Current hardware reports remain `stable_benchmark=false`. Claim Level 1 means a
qualified first sample, not stable performance or acceleration.

The repository currently makes no claim of:

- CPU acceleration or superiority;
- stable one-device performance;
- measured DRAM, NoC, PCIe, or compute bandwidth;
- energy efficiency or application speedup;
- full device-side Hamiltonian coefficient lowering;
- dual-device or aggregate N300 performance;
- arbitrary quantum-circuit simulation or quantum-hardware replacement;
- Tenstorrent endorsement.

## Primary references

- [SU2ComposeBench report](docs/benchmarks/su2-compose-bench.md)
- [Wormhole qmul report](docs/benchmarks/wormhole-qmul.md)
- [Hamiltonian roadmap](docs/hamiltonian-evolution-roadmap.md)
- [Benchmark claim policy](docs/benchmarks/claim-policy.md)
- [Documentation index](docs/index.md)
