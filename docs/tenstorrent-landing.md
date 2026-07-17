# Tenstorrent Landing Page

This page is for Tenstorrent engineers arriving from `tt-awesome`, GitHub
Discussions, or a maintainer handoff.

> **RQM runs fused time-ordered SU(2) evolution for two-level Hamiltonian simulation on Tenstorrent Wormhole.**

H1 lowers piecewise-constant two-level Hamiltonian coefficients into FP32
rotors and phase pairs on the CPU. Wormhole performs their ordered composition.
H2A device-side Hamiltonian coefficient lowering is the current hardware evidence
milestone. Its compensated single-core candidate passed the frozen nine-case
N300 contract and one non-designated pilot. A later designated device-0
session passed the same frozen contract, establishing a separate Claim Level 0
silicon-conformance release with no performance or stability claim. H1 is a real stage of a
Hamiltonian-simulation pipeline. H2B now has a CPU/reference foundation and a
two-program TT-Metal candidate source that keeps the H2A-to-H1 intermediate on
device. Contract-v1 Session 2 passed the corrected environment preflight but
did not pass due to runtime/dispatch synchronization during device
initialization. The exact boundaries are documented in
[SU2ComposeBench](benchmarks/su2-compose-bench.md).

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
persistent multicore/SFPU qmul: Claim Level 2 from three qualified device-0 sessions
SU2ComposeBench fused H1 path: Claim Level 2 stable one-device release plus retained historical v2 campaign
EntanglementDynamicsBench: CPU reference foundation only; no hardware claim or claim level
HamiltonianLoweringBench H2A: Claim Level 0 silicon conformance; stable_benchmark=false
HamiltonianEvolutionBench H2B: Contract-v1 Session 2 retained; did not pass (runtime); no claim level
```

The committed TT-Lang and tt-emule reports are simulator/emulation artifacts.
The scalar N300 report is real-hardware correctness evidence. The separate
Stage B report is performance-eligible architecture evidence. The persistent
path removes repeated process/device creation from each measurement session.

The [SU2ComposeBench report](benchmarks/su2-compose-bench.md) adds fused,
time-ordered SU(2) evolution. Its historical fused/unfused comparison remains
Level 1 and non-stable. A separately frozen fused-only v3 campaign retained
three fresh N300 cold-start sessions and passed every frozen 5% stability gate,
supporting the aggregate Claim Level 2 release with `stable_benchmark=true`.

The historical [v2 preregistration](../benchmarks/manifests/su2-compose-stability-preregistration-v2.json)
and [qualification artifact](../benchmarks/processed/wormhole-su2-compose-stability-qualification.json)
preserve the earlier non-qualifying outcome. The [v3 qualification](../benchmarks/processed/wormhole-su2-compose-v3-stability-qualification.json)
is reproducible from the designated packages. Neither release makes an
acceleration claim.

The public [Wormhole qmul benchmark report](benchmarks/wormhole-qmul.md)
packages the current evidence, deterministic charts, claim policy, provenance,
and limitations. Three independent device-0 sessions pass the preregistered
gates, supporting Claim Level 2 stable one-device performance.

The [hardware evidence report](benchmarks/wormhole-qmul-hardware-evidence.md)
separates stability evidence from device-parity, scaling, profiler, ceiling,
and saturation diagnostics.

The [EntanglementDynamicsBench foundation](benchmarks/entanglement-dynamics-bench.md)
extends the reference layer from local U(2) operations to joint two-qubit state
evolution and entanglement metrics. It is deliberately outside the hardware
evidence ladder until a separate device contract is designed and qualified.

The [H2B foundation](benchmarks/hamiltonian-evolution-h2b.md) uses the pinned
TT-Metal baseline and directly connects compensated H2A to protected fused H1
through device DRAM. It performs one input H2D and one final-output D2H with no
intermediate host transfer. Session 1 retains the original missing-runtime-root
failure. Session 2 propagated both roots and attempted all 20 frozen cases
once, but TT-Metal stalled in dispatch/mailbox synchronization and produced no
numerical outputs. It remains non-stable, performance-ineligible, and
claim-level null.

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

The persistent path completed
[persistent conformance](../reports/tt_hardware_qmul_stage_b_persistent_conformance.md)
and three independent performance sessions. The
[timing audit](../reports/tt_hardware_qmul_stage_b_persistent_timing_audit.md)
records exact synchronization boundaries. The
[stability qualification](../benchmarks/processed/wormhole-qmul-stability-qualification.json)
passes the published Level 2 gates.

Access state for RQM Technologies:

- Tenstorrent approved SSH access to an N300 host.
- The pinned TT-Metalium source build, scalar candidate, and separate Stage B
  multicore/SFPU candidate ran successfully.
- The committed report records exact repo, TT-Metalium, compiler, runtime, and
  candidate-binary provenance.

## How A Tenstorrent Engineer Can Help

The remaining engineering ask is review of the one-device multicore/SFPU
architecture, diagnostic evidence, and next optimization choices. The qmul
release supports stable one-device performance, but no acceleration or
matched-baseline claim. Placement guidance remains welcome but is independent
of the external candidate evidence.

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

Every new individual hardware sample should use `execution_label=hardware` and
`stable_benchmark=false`; only a separate aggregate qualification may promote
a release-level stability claim.

## Key Links

- [Wormhole qmul benchmark report](benchmarks/wormhole-qmul.md)
- [Wormhole qmul hardware evidence](benchmarks/wormhole-qmul-hardware-evidence.md)
- [Benchmark claim policy](benchmarks/claim-policy.md)
- [EntanglementDynamicsBench reference foundation](benchmarks/entanglement-dynamics-bench.md)
- [StructuredBench specification](structuredbench-spec.md)
- [Tenstorrent RFC](tenstorrent-rfc.md)
- [TT-Metalium qmul design](tt-metalium-qmul-design.md)
- [Functional Tenstorrent quickstart](tenstorrent-functional-quickstart.md)
- [tt-emule qmul validation plan](tt-emule-qmul-validation-plan.md)
- [tt-emule qmul candidate report](../reports/tt_emule_qmul_candidate.md)
- [TT-Lang simulator report](../reports/tt_lang_qmul_sim.md)
- [Outreach packet](../reports/tenstorrent_packet.md)
