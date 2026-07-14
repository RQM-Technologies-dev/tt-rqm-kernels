# QuantumIR Roadmap

This roadmap describes a future QuantumIR layer above the existing
`tt-rqm-kernels` foundation. Its former broad Hamiltonian phase is now refined
by the concrete [SU2HamiltonianBench roadmap](hamiltonian-evolution-roadmap.md):
CPU-lowered `SU2ComposeBench` first, device-side coefficient lowering only
after stable H1 evidence.

The current active backend path remains:

```text
StructuredBench qmul
-> TT-Lang simulator evidence
-> tt-emule candidate evidence
-> Tenstorrent Cloud or hardware report
-> maintainer-guided TT-Metalium placement
```

QuantumIR work should start only after the representation and validation rules
are clear enough to preserve the repo's current correctness-first standard.

## Phase 0: Documentation And Terminology

Goal:

- define the QuantumIR thesis without changing package code
- describe how quantum-mechanics concepts map onto existing structured tensor
  kernels
- make clear that this is a future top layer, not a replacement for qmul,
  StructuredBench, TT-Lang, tt-emule, or TT-Metalium work

Exit criteria:

- `docs/quantum-ir.md` defines the conservative thesis
- `docs/quantum-ir-operator-mapping.md` maps initial concepts to existing
  operators
- README and `plan.md` link the direction without making it the active backend
  blocker

## Phase 1: Single-Qubit SU(2) Reference

Goal:

- represent a single-qubit SU(2) gate as a unit quaternion rotor
- validate the representation against a standard 2x2 complex matrix reference
- keep all work in CPU/PyTorch reference form until the mapping is stable

Candidate checks:

- identity gate maps to the identity rotor
- composition agrees with the matrix reference within dtype tolerance
- rotor normalization keeps unit-norm error bounded
- inverse/adjoint behavior agrees with conjugate or inverse reference paths

Exit criteria:

- a small reference example and tests exist
- the docs state exactly which gate subset is represented
- no backend or hardware performance claim is made

## Phase 2: Controlled Rotations And Batched Circuits

Goal:

- extend the reference path to controlled rotations and small batched circuit
  patterns
- determine whether the StructuredBench report model needs a quantum-oriented
  suite or whether existing qmul/qrotate/phase suites are sufficient

Candidate checks:

- controlled-rotation reference compares against a direct complex matrix path
- batch dimensions follow normal PyTorch broadcasting expectations
- report labels remain CPU, simulator, emulation, or hardware according to the
  actual environment

Exit criteria:

- the workload is still small, inspectable, and correctness-tested
- no claim is made about broad quantum-circuit acceleration

## Phase 3: Hamiltonian Evolution And Spin Systems

Goal:

- implement the focused `SU2ComposeBench` contract for ordered noncommuting
  two-level evolution before considering broader spin-system workloads

Candidate checks:

- define input/output tensor contracts before writing kernels
- compare against a conventional reference implementation
- measure stability and error growth explicitly

Exit criteria:

- at least one workload has a clear reference, report shape, and conservative
  interpretation
- the workload does not depend on speculative physics claims

## Phase 4: Backend And Compiler Discussion

Goal:

- decide whether any QuantumIR-derived operator should lower to an existing
  StructuredBench kernel, a TT-NN custom operation, a TT-Metalium kernel, or a
  TT-MLIR representation

Preconditions:

- real lower-stack evidence exists for qmul beyond CPU-only reports
- any proposed fused operator has a concrete reference contract and benchmark
  case
- maintainer guidance has clarified placement expectations

Exit criteria:

- backend discussion is grounded in working code and measured reports
- TT-NN and TT-MLIR questions stay secondary until lower-stack evidence exists

## Guardrails

QuantumIR work must not:

- claim RQM replaces quantum hardware
- claim arbitrary quantum computation is efficiently classically simulable
- ask Tenstorrent for native quaternion hardware or a new chip feature
- present simulator or emulation output as hardware performance
- displace the active qmul, tt-emule, Cloud/hardware, or maintainer-placement
  work
