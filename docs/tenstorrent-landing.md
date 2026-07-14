# Tenstorrent Landing Page

This page is for Tenstorrent engineers arriving from `tt-awesome`, GitHub
Discussions, or a maintainer handoff.

> **RQM runs quantum Hamiltonian simulations on Tenstorrent.**

The first implementation executes fused, time-ordered SU(2) evolution
on Wormhole using CPU-lowered FP32 evolution operators. A later stage will
lower Hamiltonian coefficients on device. The exact preregistered boundary is
documented in [SU2ComposeBench](benchmarks/su2-compose-bench.md).

## What This Is

`tt-rqm-kernels` is an independent RQM Technologies LLC project for structured
tensor kernels represented inside ordinary floating-point tensors.

The first contract is deliberately small:

```text
qmul: float32 [N, 4] x [N, 4] -> [N, 4]
lane order: [real, i, j, k]
operation: Hamilton product
```

The goal is to make a compact benchmark class between scalar elementwise ops
and large matmul. It is useful for testing fixed cross-lane dependencies, data
movement, register reuse, fusion opportunities, arithmetic intensity, and
correctness reporting.

## Current Evidence Ladder

```text
CPU/PyTorch reference: implemented
scalar reference checks: implemented
StructuredBench report schema: implemented
TT-Lang simulator qmul: implemented, simulator-only
tt-emule TT-Metalium candidate: implemented, emulation-only
Tenstorrent N300 hardware report: Stage A conformance present
qmul integrity gate: whole-output float64 conformance and strict metrics v2
current scalar RISC-V candidate: Stage A correctness baseline, not performance-eligible
multicore/SFPU candidate: Stage B conformance and first official sweep present
persistent multicore/SFPU path: implemented; hardware qualification is separate
```

The committed TT-Lang and tt-emule reports are simulator/emulation artifacts.
The scalar N300 report is real-hardware correctness evidence. The separate
Stage B report is performance-eligible architecture evidence, but its first
sample is explicitly not stable and is not an acceleration comparison.
The persistent-device path removes repeated process/device creation from the
measurement session while preserving that non-claim.

The public [Wormhole qmul benchmark report](benchmarks/wormhole-qmul.md)
packages the current evidence, deterministic charts, claim policy, provenance,
and limitations. It classifies the single persistent session as Claim Level 1,
not as a stable result.

## Run It In 10 Minutes

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m tt_rqm_kernels.structuredbench --suite smoke --items 128 --iters 1 --warmup 0
python scripts/validate_qmul_candidate.py \
  --command "python scripts/qmul_external_reference.py" \
  --items 128 \
  --iters 1 \
  --warmup 0
```

Check the Tenstorrent-facing local state:

```bash
python scripts/repo_status.py
python scripts/rqm_tt_quickstart.py --check
```

## Current Milestone

The existing `[N, 4]` `qmul` path passed its first real N300 Stage A gate. See
[the StructuredBench report](../reports/tt_hardware_qmul_quickstart.md) and
[environment record](../reports/tt_hardware_qmul_environment.txt).

The Float32 multicore/SFPU implementation then passed its protected
[N=128 conformance gate](../reports/tt_hardware_qmul_stage_b_candidate_conformance.md),
[architecture audit](../reports/tt_hardware_qmul_stage_b_architecture_audit.md),
and [first official Stage B sweep](../reports/tt_hardware_qmul_stage_b_performance.md)
on Wormhole device 0. The first sweep keeps `stable_benchmark=false`; no
acceleration claim is made.

The next evidence rung is the separate persistent-device conformance and
three-size timing path. Its lifecycle, timing phases, and future stability
thresholds are defined in
[the preregistered methodology](stage-b-stability-methodology.md). It does not
modify the Stage A or first Stage B records and does not use device 1.

That path has now completed its first
[persistent conformance](../reports/tt_hardware_qmul_stage_b_persistent_conformance.md)
and [persistent performance](../reports/tt_hardware_qmul_stage_b_persistent_performance.md)
sessions. The [timing audit](../reports/tt_hardware_qmul_stage_b_persistent_timing_audit.md)
records exact synchronization boundaries and nonclaims. The result remains a
single-session, non-stable methodology artifact.

Access state for RQM Technologies:

- Tenstorrent approved SSH access to an N300 host.
- The pinned TT-Metalium source build, scalar candidate, and separate Stage B
  multicore/SFPU candidate ran successfully.
- The committed report records exact repo, TT-Metalium, compiler, runtime, and
  candidate-binary provenance.

## How A Tenstorrent Engineer Can Help

The remaining engineering ask is review of the one-device multicore/SFPU
architecture and its measurement scope before any stable or comparative
performance claim. Placement guidance remains welcome but is independent of
the external candidate evidence.

The copy/paste handoff is:

- [Tenstorrent engineer copy/paste packet](tenstorrent-engineer-copy-paste-packet.md)

Expected returned artifacts:

```text
reports/tt_hardware_qmul_quickstart.json
reports/tt_hardware_qmul_quickstart.md
reports/tt_hardware_qmul_stage_b_candidate_conformance.json
reports/tt_hardware_qmul_stage_b_candidate_conformance.md
reports/tt_hardware_qmul_stage_b_performance.json
reports/tt_hardware_qmul_stage_b_performance.md
reports/tt_hardware_qmul_stage_b_persistent_conformance.json
reports/tt_hardware_qmul_stage_b_persistent_conformance.md
reports/tt_hardware_qmul_stage_b_persistent_performance.json
reports/tt_hardware_qmul_stage_b_persistent_performance.md
```

The first hardware sample should use `execution_label=hardware` and
`stable_benchmark=false`.

## Key Links

- [Wormhole qmul benchmark report](benchmarks/wormhole-qmul.md)
- [Benchmark claim policy](benchmarks/claim-policy.md)
- [StructuredBench specification](structuredbench-spec.md)
- [Tenstorrent RFC](tenstorrent-rfc.md)
- [TT-Metalium qmul design](tt-metalium-qmul-design.md)
- [Functional Tenstorrent quickstart](tenstorrent-functional-quickstart.md)
- [tt-emule qmul validation plan](tt-emule-qmul-validation-plan.md)
- [tt-emule qmul candidate report](../reports/tt_emule_qmul_candidate.md)
- [TT-Lang simulator report](../reports/tt_lang_qmul_sim.md)
- [Outreach packet](../reports/tenstorrent_packet.md)
