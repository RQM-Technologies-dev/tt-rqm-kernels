# Tenstorrent Functional Quickstart

This quickstart is for engineers who want to run the current Tenstorrent-facing
`qmul` path in `tt-rqm-kernels`.

`tt-rqm-kernels` is an independent RQM Technologies LLC project. It is not an
official Tenstorrent repository unless and until accepted or co-developed by
Tenstorrent.

## What Runs Today

Current runnable paths:

- CPU/PyTorch reference kernels for quaternion, rotor, inverse, normalization,
  and phase workloads
- StructuredBench CPU/PyTorch reports
- optional TT-Lang functional simulator `qmul` reports
- external-qmul candidate harness
- experimental TT-Metalium scalar RISC-V `qmul` candidate staged outside
  upstream `tt-metal`
- tt-emule/Docker wrapper for that candidate when the local `tt-metal`,
  `tt-emule`, Docker, and built candidate binary are available

Not implemented yet:

- official TT-NN operator
- upstream TT-Metalium example placement
- stable Tenstorrent hardware benchmark
- native quaternion datatype or new hardware feature

## Execution Labels

StructuredBench reports use explicit execution labels:

```text
cpu        CPU/PyTorch reference or CPU external candidate
simulator  TT-Lang functional simulator only
emulation  tt-emule or another emulator path
hardware   real Tenstorrent Cloud or hardware execution only
```

Never label TT-Lang simulator or tt-emule output as hardware.

## Quickstart Check

Run:

```bash
python scripts/rqm_tt_quickstart.py --check
```

The check prints:

- Python package import status
- torch availability
- repo root
- report output directory
- `TT_METAL_HOME` or sibling `tt-metal` checkout status
- `TT_EMULE_HOME` or sibling `tt-emule` checkout status
- Docker availability
- emule candidate script presence
- emule candidate binary presence
- hardware command configuration

The check does not run a kernel and does not claim hardware readiness.

## CPU/PyTorch Reference

Run the smoke suite:

```bash
python -m tt_rqm_kernels.structuredbench --suite smoke
```

Run the focused `qmul` suite:

```bash
python -m tt_rqm_kernels.structuredbench \
  --suite qmul \
  --items 128 \
  --iters 1 \
  --warmup 0
```

These are CPU/PyTorch reference outputs unless another backend is explicitly
selected.

## TT-Lang Simulator

Check simulator availability:

```bash
python scripts/run_ttlang_qmul_smoke.py --check
```

Run a simulator qmul smoke when `tt-lang-sim` is installed:

```bash
python -m tt_rqm_kernels.structuredbench \
  --backend tt-lang-sim \
  --suite qmul \
  --items 128 \
  --iters 1 \
  --warmup 0
```

TT-Lang output must use `execution_label=simulator`. It is useful for functional
validation and trace/stat diagnostics, not hardware performance.

## tt-emule Mode

When the local environment has `tt-metal`, `tt-emule`, Docker, and the built
experimental qmul candidate binary, run:

```bash
python scripts/rqm_tt_quickstart.py --mode emule
```

By default this uses:

```text
bash experimental/tt_metalium_qmul/run_candidate_docker.sh
```

Default reports are written to:

```text
reports/tt_emule_qmul_quickstart.json
reports/tt_emule_qmul_quickstart.md
```

This path must use `execution_label=emulation`. It is not hardware performance.

## Hardware Mode

Hardware mode requires a real Tenstorrent Cloud or hardware command that
implements the external-qmul protocol.

In Tenstorrent Console, the observed custom-code path starts at:

```text
Compute -> Resources -> Request Capacity
```

Request capacity for one small `[N, 4]` StructuredBench `qmul` hardware report.
After access is granted, run either in a managed VSCode/browser instance or an
SSH baremetal shell.

Configure it with either:

```bash
export TT_RQM_HARDWARE_QMUL_COMMAND="/path/to/real_hardware_qmul_candidate"
python scripts/rqm_tt_quickstart.py --mode hardware
```

or:

```bash
python scripts/rqm_tt_quickstart.py \
  --mode hardware \
  --command "/path/to/real_hardware_qmul_candidate"
```

Default reports are written to:

```text
reports/tt_hardware_qmul_quickstart.json
reports/tt_hardware_qmul_quickstart.md
```

Hardware mode fails clearly if no hardware command is configured. Do not pass
the tt-emule Docker wrapper to hardware mode.

## Direct qmul Example

The developer example generates deterministic `[N, 4]` quaternion inputs, runs
CPU/PyTorch `qmul`, and optionally compares a Tenstorrent external-qmul command:

```bash
python examples/tenstorrent_qmul_quickstart.py --items 128
```

With emulation or hardware:

```bash
python examples/tenstorrent_qmul_quickstart.py \
  --mode emule \
  --items 128 \
  --iters 1 \
  --warmup 0
```

The example prints max absolute error, RMS error, checksum, latency, and
throughput when an external path is configured.

## external-qmul Protocol

The external candidate command reads:

```text
a.bin
b.bin
manifest.json
```

and writes:

```text
out.bin
metrics.json
```

The supported contract is:

```text
float32 [N, 4] x [N, 4] -> [N, 4]
lane order: [real, i, j, k]
operation: Hamilton product
```

StructuredBench compares the candidate output against CPU/PyTorch and scalar
references, then reports latency, throughput, numerical error, FLOPs/sec,
effective GB/sec, arithmetic intensity, and checksum.

## Interpretation

This quickstart makes the current Tenstorrent-facing path runnable and
inspectable. It does not make `tt-rqm-kernels` an official Tenstorrent project,
does not request a native quaternion datatype, and does not request any new
hardware feature.
