# Tenstorrent Landing Page

This page is for Tenstorrent engineers arriving from `tt-awesome`, GitHub
Discussions, or a maintainer handoff.

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
```

The committed TT-Lang and tt-emule reports are simulator/emulation artifacts.
The N300 report is real-hardware correctness evidence, but it is explicitly not
a stable or performance-eligible benchmark.

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

The next accelerator milestone is a multicore/SFPU Stage B implementation that
can qualify for the repository's performance methodology. No performance claim
is made from the scalar Stage A result.

Access state for RQM Technologies:

- Tenstorrent approved SSH access to an N300 host.
- The pinned TT-Metalium source build and scalar candidate ran successfully.
- The committed report records exact repo, TT-Metalium, compiler, runtime, and
  candidate-binary provenance.

## How A Tenstorrent Engineer Can Help

The narrow ask is to run the existing candidate in a real Tenstorrent
environment and return one hardware-labeled StructuredBench report. Placement
guidance is welcome if offered, but the active validation plan does not depend
on it.

The copy/paste handoff is:

- [Tenstorrent engineer copy/paste packet](tenstorrent-engineer-copy-paste-packet.md)

Expected returned artifacts:

```text
reports/tt_hardware_qmul_quickstart.json
reports/tt_hardware_qmul_quickstart.md
```

The first hardware sample should use `execution_label=hardware` and
`stable_benchmark=false`.

## Key Links

- [StructuredBench specification](structuredbench-spec.md)
- [Tenstorrent RFC](tenstorrent-rfc.md)
- [TT-Metalium qmul design](tt-metalium-qmul-design.md)
- [Functional Tenstorrent quickstart](tenstorrent-functional-quickstart.md)
- [tt-emule qmul validation plan](tt-emule-qmul-validation-plan.md)
- [tt-emule qmul candidate report](../reports/tt_emule_qmul_candidate.md)
- [TT-Lang simulator report](../reports/tt_lang_qmul_sim.md)
- [Outreach packet](../reports/tenstorrent_packet.md)
