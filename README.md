# tt-rqm-kernels

[![CI](https://github.com/RQM-Technologies-dev/tt-rqm-kernels/actions/workflows/ci.yml/badge.svg)](https://github.com/RQM-Technologies-dev/tt-rqm-kernels/actions/workflows/ci.yml)

`tt-rqm-kernels` is a hardware-validated TT-Metalium kernel path for structured
FP32 tensor math on Tenstorrent hardware. Quaternions, rotors, complex phase
pairs, and SU(2) state remain ordinary tensors; the structure comes from their
lane conventions and the operators applied to them.

This is an independent open-source RQM Technologies LLC project. It is not an
official Tenstorrent repository or a statement of Tenstorrent endorsement.

> **RQM runs fused time-ordered SU(2) evolution for two-level Hamiltonian simulation on Tenstorrent Wormhole.**

H1 lowers piecewise-constant two-level Hamiltonian coefficients into FP32
rotors and phase pairs on the CPU. Wormhole performs their ordered composition.
H2A device-side Hamiltonian coefficient lowering is now the active technical
milestone. A distinct compensated single-core candidate retains FP32 angle
high/low components through device-side split-period reduction and passes the
frozen nine-case N300 contract. Its one retained pilot is non-designated and
qualification-ineligible, so no H2A claim level or hardware conformance release
exists. H1 is a real stage of the simulation pipeline, not the complete
device-side pipeline.

## For Tenstorrent engineers

This repository is designed to be a compact, reproducible structured-kernel
example rather than a request for a new datatype or hardware feature.

- **Why it maps:** `qmul` has fixed four-lane cross-dependencies,
  noncommutative ordering, explicit data movement, multicore Tensix
  compute/SFPU arithmetic, and concrete opportunities for register/L1 reuse
  and fusion.
- **What is proven:** the one-device `qmul` release and fused-only
  `SU2ComposeBench` release each passed three qualified N300 sessions. The
  historical SU2 fused/unfused campaign remains non-qualifying.
- **The current engineering decisions:** the SU2 profiler did not isolate a
  semantics-preserving fusion/layout correction, while the separate qmul
  placement question remains a design discussion rather than an
  implementation request.

That placement decision is now tracked in
[tenstorrent/tt-metal#49887](https://github.com/tenstorrent/tt-metal/issues/49887).
No upstream implementation PR will be opened until maintainers indicate whether
the minimal `qmul` path belongs as a TT-Metalium programming example or an
experimental TT-NN `ProgramDescriptor` operation.

The evidence is intentionally separated from broader application claims. The
[Wormhole qmul report](docs/benchmarks/wormhole-qmul.md) records the qualified
one-device result; [SU2ComposeBench](docs/benchmarks/su2-compose-bench.md)
records the current fused-only Level 2 composition evidence.

## Current proven result

The repository has two public N300 device-0 benchmark releases: qmul and
`SU2ComposeBench` each have a three-session stable one-device release.
`SU2ComposeBench` is fused-only; neither result is an acceleration claim.

| Evidence | Implementation | Claim | Stable benchmark |
|---|---|---|---|
| qmul | multicore Tensix compute/SFPU on one Wormhole device; Stage A baseline retained | Level 2 | `true` |
| SU2ComposeBench | fused time-ordered SU(2) composition on one Wormhole device | Level 2 | `true` |

The individual qmul and SU2 source-session reports remain
`stable_benchmark=false`; each aggregate release is `true` only because three
qualified designated sessions passed. See the [Wormhole qmul report](docs/benchmarks/wormhole-qmul.md)
and [SU2ComposeBench report](docs/benchmarks/su2-compose-bench.md) for the
exact contracts, results, and limitations.

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

## Why it matters

Structured kernels occupy useful territory between scalar elementwise code and
large matrix multiplication:

- fixed cross-lane dependencies test more than independent elementwise math;
- noncommutative ordered composition makes execution order observable;
- register and L1 reuse create concrete fusion opportunities;
- fused chains can avoid intermediate DRAM movement;
- the same representation supports disciplined scientific-computing and
  physical-AI experiments without introducing a native quaternion datatype.

These results do not establish CPU acceleration, stable fused/unfused SU2
comparison, measured hardware bandwidth, energy efficiency, application
speedup, full device-side Hamiltonian lowering, dual-device scaling, or
Tenstorrent endorsement.

## What you can use `qmul` for

`qmul(a, b)` is the batched Hamilton product of ordinary floating-point
quaternion tensors with final lane layout `[real, i, j, k]`. It is useful when
each four-lane value represents a structured transformation and the order of
composition matters:

- **Rotation and pose pipelines:** compose unit quaternions that encode 3D
  orientations, then fuse normalization, conjugation, or vector rotation into
  a larger tensor program.
- **Two-level Hamiltonian simulation:** compose the unit-quaternion rotors and
  phase pairs produced by CPU lowering of piecewise-constant Hamiltonian
  coefficients. This is the H1 stage implemented by `SU2ComposeBench`.
- **Structured-kernel development:** use the compact `[N, 4]` contract as a
  reproducible target for testing cross-lane arithmetic, noncommutative order,
  layout choices, and fusion on a Tenstorrent backend.

The reference operator supports normal PyTorch broadcasting over leading
dimensions, so it can serve as a building block inside batched simulations and
tensor pipelines rather than only as a standalone benchmark. See the
[operator contract](docs/operator-contracts.md#qmul) for its exact equation,
shapes, and SU(2) matrix convention.

The present device result proves the `qmul` kernel contract and its
one-device benchmark stability; it does not yet prove end-to-end speedup for a
rotation, scientific-simulation, or physical-AI application. Device-side
Hamiltonian coefficient lowering, broader physical-AI benchmarks, and
two-qubit execution remain future work.

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
