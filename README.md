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

The repository contains real N300 device-0 evidence for quaternion
multiplication and `SU2ComposeBench`. Three independent persistent qmul
sessions support a stable one-device release; the SU2 comparison remains a
single non-stable session. Neither result is an acceleration claim.

| Evidence | Implementation | Claim | Stable benchmark |
|---|---|---|---|
| Stage A qmul conformance | scalar RISC-V correctness baseline | Level 0 | `false` |
| Stage B qmul | multicore Tensix compute/SFPU | Level 1 | `false` |
| Persistent qmul | three qualified device-0 sessions | Level 2 | `true` |
| SU2ComposeBench H1 conformance | fused and unfused ordered composition | Level 0 | `false` |
| SU2ComposeBench H1 comparison | one qualified fused/unfused session | Level 1 | `false` |

The SU(2) comparison validates every output against independent CPU oracles.
The fused path retains rotor and phase accumulators in Tensix L1; the unfused
path uses repeated qmul-plus-phase dispatches with DRAM ping-pong storage. See
the [SU2ComposeBench report](docs/benchmarks/su2-compose-bench.md) for the exact
contract, results, and limitations.

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
python scripts/validate_su2_compose_release.py
python scripts/reproduce_wormhole_su2_compose.py --check
```

## Documentation paths

- [Architecture and operator contracts](docs/operator-contracts.md): tensor
  conventions, Hamilton products, rotor/phase operators, and backend designs.
- [Benchmark evidence and reproduction](docs/benchmarks/index.md): qmul and
  SU2ComposeBench reports, release manifests, claim policy, and methodology.
- [Roadmap and future work](plan.md): proven capabilities, active evidence
  work, deferred integrations, H2, and non-goals.

The [documentation index](docs/index.md) organizes secondary runbooks,
simulator/emulation material, cloud-access history, outreach packets,
QuantumIR notes, and longer-term physical-AI directions.

## License

Apache License 2.0. See [LICENSE](LICENSE).
