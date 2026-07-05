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
Tenstorrent hardware report: not implemented yet
```

The committed TT-Lang and tt-emule reports are validation artifacts. They are
not hardware performance claims.

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

## Current Blocker

The next useful evidence target is one real Tenstorrent hardware run of the
existing `[N, 4]` `qmul` path.

Observed Console state for RQM Technologies:

- API inference, Usage, Billing, Compute, and Resources are visible.
- No dedicated hardware allocation is assumed.
- Instances and Baremetal are blocked until access is granted.
- `Compute -> Resources -> Request Capacity` opens, but `Resource Type` has no
  selectable option for the org, so a capacity request cannot currently be
  submitted from the Console.

## How A Tenstorrent Engineer Can Help

The narrow ask is one of:

1. advise where a minimal `[N, 4]` TT-Metalium `qmul` example should live, or
2. run the existing candidate in a real Tenstorrent environment and return one
   hardware-labeled StructuredBench report.

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
