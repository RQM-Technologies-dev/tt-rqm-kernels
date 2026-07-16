# tt-rqm-kernels

[![CI](https://github.com/RQM-Technologies-dev/tt-rqm-kernels/actions/workflows/ci.yml/badge.svg)](https://github.com/RQM-Technologies-dev/tt-rqm-kernels/actions/workflows/ci.yml)

`tt-rqm-kernels` is RQM Technologies LLC's hardware-validated laboratory for
running quaternionic rotation, phase, and Hamiltonian-evolution mathematics as
structured FP32 tensor workloads on Tenstorrent accelerators. Quaternions,
rotors, complex phase pairs, and SU(2) state are represented as ordinary
floating-point tensors; no native quaternion datatype is required.

This is an independent open-source RQM Technologies LLC project. It is not an
official Tenstorrent repository or a statement of Tenstorrent endorsement.

> **RQM runs fused time-ordered SU(2) evolution for two-level Hamiltonian simulation on Tenstorrent Wormhole.**

H1 consumes rotor and phase steps pre-lowered on the CPU; Wormhole composes
them in exact time order.

## What this repository is

The repository turns RQM mathematical operations into reproducible CPU
references, TT-Metalium kernels, tests, benchmark contracts, and real
Wormhole hardware evidence. It demonstrates a useful workload class between
independent scalar or elementwise operations and large matrix multiplication.

These are small, structured tensor programs: values have fixed cross-lane
dependencies, operation order is noncommutative, and rotations and phases must
be composed in the right sequence. That makes register and L1 reuse, fused
chains, and reduced intermediate memory movement visible engineering choices
instead of abstract optimizations.

## What is already here

| Component | What runs on Wormhole | Current protected result |
|---|---|---|
| Quaternion multiplication — `qmul` | Batched Hamilton products over ordinary FP32 `[N,4]` tensors in `[real, i, j, k]` order, for structured rotations and transformations where order matters | Claim Level 2 stable one-device performance from three qualified N300 sessions |
| Fused time-ordered SU(2) composition — H1 / `SU2ComposeBench` | Exact-time-order composition of pre-lowered rotor and phase steps; a real stage of two-level Hamiltonian simulation | Claim Level 2 fused-only stable one-device performance from three qualified v3 sessions |
| Device-side Hamiltonian coefficient lowering — H2A | Converts two-level Hamiltonian coefficients and `dt` into the rotor and phase steps consumed by H1 | Claim Level 0 silicon conformance from one designated N300 session; `stable_benchmark=false`, `performance_eligible=false` |

The H1 result does not establish a stable fused/unfused comparison or an
acceleration claim. H2A is a completed hardware-conformance milestone, not
future work, but it is not a performance or stability result.

## What it can be used for

### Two-level Hamiltonian simulation

The strongest current application path is a reproducible two-level evolution
pipeline:

```text
Hamiltonian coefficients
        ↓
H2A device-side lowering
        ↓
rotors and phase pairs
        ↓
H1 time-ordered composition
        ↓
final U(2) evolution operator
```

This supports development and study of qubit and spin-½ evolution, driven
two-level systems, quantum-control and pulse-sequence simulation, and
time-dependent SU(2) dynamics. It does not claim arbitrary quantum-circuit
simulation or quantum-hardware execution.

### Rotation, pose, and physical-state workloads

`qmul` is also a reusable contract for ordered 3D rotation pipelines such as
robot or drone orientation, inertial-navigation updates, camera or vehicle
pose, satellite attitude, and quaternion-based sensor fusion. These are
enabled workload directions, not demonstrated end-to-end application
acceleration results.

### Structured-kernel research on Tenstorrent

The repository gives Tenstorrent developers a compact, reproducible workload
for cross-lane tensor arithmetic, component-planar layouts, multicore Tensix
work distribution, compute/SFPU placement, register and L1 reuse, kernel
fusion, intermediate DRAM avoidance, and whole-output numerical validation.

## What this is building toward

H2B has not begun. It would make the first device-resident two-level
Hamiltonian-evolution pipeline in this project by feeding H2A directly into
fused H1 composition, without a host round trip for intermediate rotor and
phase tensors:

```text
Hamiltonian coefficients + dt
        ↓
device-resident H2A lowering
        ↓
directly into fused H1 composition
        ↓
final rotor and phase
```

H2B must earn separate evidence and cannot inherit H1 or H2A claim status. The
broader purpose is to move RQM mathematics from theory and Python references
into inspectable, reproducible, hardware-backed Tenstorrent workloads that
outside engineers can evaluate and potentially integrate.

## Why Tenstorrent

The workload maps naturally to ordinary FP32 tensors, fixed structured
arithmetic, visible noncommutative order, explicit data movement, and
multicore Tensix compute/SFPU implementation. It offers practical fusion and
local-memory reuse opportunities without asking for a new datatype or hardware
feature.

The open placement question is tracked in
[tenstorrent/tt-metal#49887](https://github.com/tenstorrent/tt-metal/issues/49887):
whether the minimal `qmul` path belongs as a TT-Metalium programming example or
an experimental TT-NN `ProgramDescriptor` operation. Tenstorrent has not
accepted either placement.

For detailed evidence, see the [Wormhole qmul report](docs/benchmarks/wormhole-qmul.md),
[SU2ComposeBench report](docs/benchmarks/su2-compose-bench.md), and
[H2A report](docs/benchmarks/hamiltonian-lowering-h2a.md).

## Current proven result

The repository has three public N300 device-0 benchmark releases. qmul and
`SU2ComposeBench` each have a three-session stable one-device release;
HamiltonianLoweringBench H2A has a separate one-session Claim Level 0
silicon-conformance release. `SU2ComposeBench` is fused-only; none is an
acceleration claim.

| Evidence | Implementation | Claim | Stable benchmark |
|---|---|---|---|
| qmul | multicore Tensix compute/SFPU on one Wormhole device; Stage A baseline retained | Level 2 | `true` |
| SU2ComposeBench | fused time-ordered SU(2) composition on one Wormhole device | Level 2 | `true` |
| HamiltonianLoweringBench H2A | device-side Hamiltonian coefficient lowering on one Wormhole device | Level 0 | `false` |

The individual qmul and SU2 source-session reports remain
`stable_benchmark=false`; each aggregate release is `true` only because three
qualified designated sessions passed. See the [Wormhole qmul report](docs/benchmarks/wormhole-qmul.md)
and [SU2ComposeBench report](docs/benchmarks/su2-compose-bench.md) for the
exact contracts, results, and limitations. The H2A source session and every
case report also remain `stable_benchmark=false` and performance-ineligible.

### SU2 stability status

SU2ComposeBench now has a **Claim Level 2** fused-only one-device result with
`stable_benchmark=true`. Three designated v3 cold-start N300 sessions passed
the hash-bound candidate/source/runtime, host, cache, lifecycle, correctness,
raw-duration, and 5% fused-stability gates. Each individual session remains
`stable_benchmark=false`; the aggregate makes no fused/unfused comparison or
acceleration claim.

The original [stability preregistration](benchmarks/manifests/su2-compose-stability-preregistration.json)
remains historical. After profiler review retained candidate `54b91b…`, the
[new v2 preregistration](benchmarks/manifests/su2-compose-stability-preregistration-v2.json)
froze its exact candidate, source, runtime, input hashes, case order, and gates
before designated session 1. The [retained qualification result](benchmarks/processed/wormhole-su2-compose-stability-qualification.json)
records the rejected dispersion and cross-session gates. The collector and
deterministic qualifier fail closed on candidate, environment, session,
correctness, or timing inconsistencies.

### Separate candidate experiment

A separate, hash-preserved N300 device-0 experiment evaluated a newly rebuilt
SU2 candidate (`54b91b…`) with two conformance cases and eight paired
performance cases. It retained two warmup pairs and ten measured pairs per
case, passed the recorded correctness checks, and remains
`stable_benchmark=false`. It is **not** a designated session in either
stability campaign. See the
[candidate experiment record](docs/benchmarks/su2-compose-candidate-54b91b.md)
and its retained raw evidence. It served only as disclosed threshold
calibration for v2. Three fresh sessions were collected after the new contract
was frozen, and their non-qualifying outcome is retained separately.

The sibling `TwoQubitHamiltonianBench` now has a CPU-only
[EntanglementDynamicsBench reference foundation](docs/benchmarks/entanglement-dynamics-bench.md).
It adds joint-state evolution and entanglement diagnostics, but has no
Tenstorrent implementation, hardware evidence, performance claim, or claim
level.

## Evidence boundaries

The present releases prove the `qmul` kernel contract, fused H1 composition,
and H2A silicon conformance within their separate contracts. They do not prove
end-to-end speedup for rotation, scientific-simulation, or physical-AI
applications. They also do not establish CPU acceleration, a stable
fused/unfused SU2 comparison, measured hardware bandwidth, energy efficiency,
dual-device scaling, a complete H2B pipeline, or Tenstorrent endorsement.

H2B integration, broader physical-AI application benchmarks, and two-qubit
hardware execution remain future work. See the
[operator contract](docs/operator-contracts.md#qmul) for exact tensor shapes,
equations, and SU(2) conventions.

## Five-minute local quickstart

The default developer path is hardware-free:

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m tt_rqm_kernels.structuredbench --suite smoke --items 128 --iters 1 --warmup 0
python scripts/repo_status.py
```

Hardware, TT-Lang, and tt-emule integrations are optional and are not required
for the local suite.

## Reproduce committed evidence

These commands validate hashes, provenance, claim gates, raw samples, and
deterministic generated outputs without accessing hardware:

```bash
python scripts/validate_benchmark_release.py
python scripts/reproduce_wormhole_qmul.py --check
python scripts/validate_su2_compose_preregistration.py
python scripts/validate_su2_compose_stability_preregistration.py
python scripts/validate_su2_compose_release.py
python scripts/reproduce_wormhole_su2_compose.py --check
python scripts/validate_entanglement_dynamics_preregistration.py
python scripts/validate_hamiltonian_lowering_preregistration.py
python scripts/validate_hamiltonian_lowering_release.py
python scripts/reproduce_wormhole_hamiltonian_lowering.py --check
python scripts/validate_repository_claims.py
```

## Documentation paths

- [Architecture and operator contracts](docs/operator-contracts.md): tensor
  conventions, Hamilton products, rotor/phase operators, and backend designs.
- [Benchmark evidence and reproduction](docs/benchmarks/index.md): qmul and
  SU2ComposeBench evidence plus the EntanglementDynamicsBench reference
  foundation.
- [Roadmap and future work](plan.md): proven capabilities, active evidence
  work, deferred integrations, H2, and non-goals.

The [documentation index](docs/index.md) organizes secondary runbooks,
simulator/emulation material, cloud-access history, outreach packets,
QuantumIR notes, and longer-term physical-AI directions.

## License

Apache License 2.0. See [LICENSE](LICENSE).
