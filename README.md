# tt-rqm-kernels

[![CI](https://github.com/RQM-Technologies-dev/tt-rqm-kernels/actions/workflows/ci.yml/badge.svg)](https://github.com/RQM-Technologies-dev/tt-rqm-kernels/actions/workflows/ci.yml)

`tt-rqm-kernels` develops structured floating-point tensor kernels for
Tenstorrent hardware. Quaternions, rotors, complex phase pairs, and SU(2) state
remain ordinary tensors; the structure comes from their lane conventions and
the operators applied to them.

This is an independent open-source RQM Technologies LLC project. It is not an
official Tenstorrent repository or a statement of Tenstorrent endorsement.

> **RQM runs fused time-ordered SU(2) evolution for two-level Hamiltonian simulation on Tenstorrent Wormhole.**

H1 lowers piecewise-constant two-level Hamiltonian coefficients into FP32
rotors and phase pairs on the CPU. Wormhole performs their ordered composition.
H2 will address device-side Hamiltonian coefficient lowering. H1 is a real
stage of a Hamiltonian-simulation pipeline, not the complete device-side
pipeline.

## Current proven result

The repository has two public N300 device-0 benchmark releases. qmul has a
three-session stable one-device release; `SU2ComposeBench` has one qualified
fused/unfused comparison session. Neither result is an acceleration claim.

| Evidence | Implementation | Claim | Stable benchmark |
|---|---|---|---|
| qmul | multicore Tensix compute/SFPU on one Wormhole device; Stage A baseline retained | Level 2 | `true` |
| SU2ComposeBench H1 | fused and unfused time-ordered SU(2) composition on one Wormhole device | Level 1 | `false` |

The individual Stage A and first Stage B qmul reports remain
`stable_benchmark=false`; qmul's aggregate release is `true` because three
qualified sessions passed. SU2 has one qualified comparison session and remains
`false`. See the [Wormhole qmul report](docs/benchmarks/wormhole-qmul.md) and
[SU2ComposeBench report](docs/benchmarks/su2-compose-bench.md) for the exact
contracts, results, and limitations.

### SU2 stability status

The published SU2 evidence remains **Claim Level 1** with
`stable_benchmark=false`. Its Level 2 qualification contract is now frozen:
three independent cold-start sessions must use the same candidate and
environment, retain every designated run, pass whole-output correctness, and
satisfy preregistered per-path and fused/unfused comparison-variability gates
across all eight benchmark cases. Sessions 2 and 3 have not been collected, so
the repository makes no stable SU2 performance claim.

The [stability methodology](docs/su2-stability-methodology.md) and
[machine-readable preregistration](benchmarks/manifests/su2-compose-stability-preregistration.json)
define the gates. The collector and deterministic qualifier fail closed on
candidate, environment, session, correctness, or timing inconsistencies.

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

These results do not establish CPU acceleration, stable SU2 performance,
measured hardware bandwidth, energy efficiency, application speedup, full
device-side Hamiltonian lowering, dual-device scaling, or Tenstorrent
endorsement.

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
