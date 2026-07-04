# Tenstorrent Execution Runbook

This runbook prepares `tt-rqm-kernels` for a future real Tenstorrent execution
of the StructuredBench `qmul` workload. It does not claim hardware performance
and it does not add TT-Metalium source code.

Use this when a Tenstorrent Cloud or local TT-Metalium environment is available
and a real `qmul` candidate executable exists.

## Status

Current local status:

- CPU/PyTorch reference path is implemented.
- Scalar spot checks are implemented.
- TT-Lang simulator `qmul` report is available and labeled simulator-only.
- External `qmul` candidate harness is implemented.
- TT-Metalium candidate package exists as a scaffold only.
- tt-emule validation preflight exists as a scaffold only.
- Real TT-Metalium host/kernel source is not implemented yet.

Placement guidance is tracked separately in the public `tt-metal` issue and the
repo-local tracker. Do not open an upstream Tenstorrent PR from this repo until
that guidance is clear.

## Official Setup References

Use the current Tenstorrent documentation as the setup source of truth:

- Tenstorrent software stack overview:
  <https://docs.tenstorrent.com/getting-started/tt-software-stack.html>
- TT-Metalium getting started:
  <https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/get_started/get_started.html>
- TT-Metalium support:
  <https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/resources/support.html>
- Tenstorrent Cloud:
  <https://tenstorrent.com/en/hardware/cloud>
- tt-emule:
  <https://github.com/tenstorrent/tt-emule>

This repo only records the StructuredBench-side commands and reporting
requirements.

## Required Environment Record

Before publishing any result, record:

```text
execution_label: cpu | simulator | emulation | hardware
repo_commit: <git rev-parse HEAD>
tt_metal_commit: <git rev-parse HEAD inside tt-metal checkout, if applicable>
tt_metal_home: <TT_METAL_HOME or TT_METALIUM_HOME>
device_type: <for example Wormhole, Blackhole, or cloud instance label>
device_count: <integer>
cloud_or_host: <local host, remote host, or Tenstorrent Cloud session>
python_version: <python --version>
torch_version: <python -c "import torch; print(torch.__version__)">
candidate_command: <exact command passed to external-qmul>
seed: <integer>
items: <integer>
iterations: <integer>
warmup: <integer>
dtype: float32
```

If a tool such as `tt-smi` is available in the environment, include its device
summary output in notes. Do not require that tool for CPU-only or simulator-only
runs.

## Local Preflight

Check whether a TT-Metalium SDK checkout is visible:

```bash
python experimental/tt_metalium_qmul/check_environment.py
```

Expected result without SDK access:

```text
TT-Metalium SDK unavailable: set TT_METAL_HOME or TT_METALIUM_HOME to a local tt-metal checkout.
```

When an SDK checkout exists:

```bash
export TT_METAL_HOME=/path/to/tt-metal
python experimental/tt_metalium_qmul/check_environment.py
```

This preflight only checks for a plausible checkout. It does not prove hardware
availability and it does not run a kernel.

## CPU/PyTorch Reference

Run a small deterministic CPU/PyTorch reference:

```bash
python -m tt_rqm_kernels.structuredbench \
  --suite qmul \
  --items 128 \
  --iters 1 \
  --warmup 0 \
  --seed 0 \
  --json-output reports/qmul_cpu_reference.json \
  --markdown-output reports/qmul_cpu_reference.md
```

Label this result:

```text
execution_label = cpu
backend = torch
device = cpu
```

## TT-Lang Simulator Check

If `tt-lang-sim` is installed, run:

```bash
python scripts/run_ttlang_qmul_smoke.py --check

python scripts/run_ttlang_qmul_smoke.py \
  --items 128 \
  --iters 1 \
  --warmup 0 \
  --seed 0 \
  --json-output reports/tt_lang_qmul_sim.json \
  --markdown-output reports/tt_lang_qmul_sim.md
```

Label this result:

```text
execution_label = simulator
backend = tt-lang-sim
device = functional-simulator
```

Simulator output validates kernel logic and report shape. It is not Tenstorrent
hardware execution.

## tt-emule Preflight

`tt-emule` is the intended emulation step before a real Tenstorrent Cloud or
hardware run. It must be labeled as emulation, not hardware.

Check whether the local environment has a plausible x86-64 Linux tt-metal plus
tt-emule checkout:

```bash
python experimental/tt_emule_qmul/check_environment.py
```

Expected local result on unsupported platforms or without checkouts:

```text
tt-emule environment unavailable: ...
```

When the source trees exist:

```bash
export TT_METAL_HOME=/path/to/tt-metal
export TT_EMULE_HOME=/path/to/tt-emule
python experimental/tt_emule_qmul/check_environment.py
```

The detailed emulation plan is in
`docs/tt-emule-qmul-validation-plan.md`. Passing this preflight does not prove
that a candidate builds or runs.

## Future TT-Metalium Candidate

The future TT-Metalium executable must satisfy the `external-qmul` protocol:

```text
[N, 4] x [N, 4] -> [N, 4]
```

StructuredBench provides:

```text
TT_RQM_EXTERNAL_QMUL_DIR
TT_RQM_EXTERNAL_QMUL_MANIFEST
a.bin
b.bin
manifest.json
```

The candidate must write:

```text
out.bin
metrics.json
```

Run the validation wrapper:

```bash
python experimental/tt_metalium_qmul/validate_candidate.py \
  --candidate-command "/path/to/tt_metalium_qmul_candidate" \
  --items 128 \
  --iters 1 \
  --warmup 0 \
  --seed 0 \
  --json-output reports/tt_metalium_qmul_candidate.json \
  --markdown-output reports/tt_metalium_qmul_candidate.md
```

For a larger report after the small run passes:

```bash
python experimental/tt_metalium_qmul/validate_candidate.py \
  --candidate-command "/path/to/tt_metalium_qmul_candidate" \
  --items 4096 \
  --iters 10 \
  --warmup 2 \
  --seed 0 \
  --json-output reports/tt_metalium_qmul_candidate_4096.json \
  --markdown-output reports/tt_metalium_qmul_candidate_4096.md
```

Label this result as either:

```text
execution_label = emulation
```

or:

```text
execution_label = hardware
```

The label must match the actual execution environment. Do not label an emulator,
simulator, CPU command, or placeholder as hardware.

## Required Report Fields

Every comparable report should include:

- schema
- backend
- device
- dtype
- seed
- items
- iterations
- warmup
- latency
- throughput
- max absolute error
- max relative error
- RMS error
- scalar reference error
- estimated FLOPs/sec
- effective GB/sec
- arithmetic intensity
- checksum

For first hardware samples, include the note:

```text
This is an initial hardware sample for methodology validation. It is not a stable performance benchmark.
```

## Publication Gate

Before publishing a hardware report:

1. Save JSON and Markdown reports.
2. Save exact commands.
3. Save repo commit and SDK commit.
4. Identify CPU, simulator, emulation, and hardware results explicitly.
5. Confirm scalar reference error is below the float32 tolerance.
6. Confirm no report text claims stable performance from a single sample.
7. Confirm no report implies Tenstorrent endorsement.

## Non-Goals

- Do not add TT-Metalium source code from this runbook.
- Do not claim hardware performance from CPU, simulator, or emulation output.
- Do not compare simulator output as if it were hardware.
- Do not frame this as defense-first.
- Do not open an upstream `tt-metal` PR until placement guidance is clear.
