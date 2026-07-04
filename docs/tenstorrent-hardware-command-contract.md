# Tenstorrent Hardware Command Contract

This contract defines the command boundary for any real Tenstorrent Cloud or
device `qmul` candidate used by `tt-rqm-kernels`.

The command is invoked by the existing `external-qmul` protocol. It must read
inputs from the working directory named by environment variables and write the
expected outputs back to that same directory.

Real hardware reports must use:

```text
execution_label=hardware
```

Do not use `execution_label=hardware` for CPU, TT-Lang simulator, tt-emule, or
other emulation output.

## Environment Inputs

The runner sets:

```text
TT_RQM_EXTERNAL_QMUL_DIR
TT_RQM_EXTERNAL_QMUL_MANIFEST
```

`TT_RQM_EXTERNAL_QMUL_DIR` points to a temporary work directory.

`TT_RQM_EXTERNAL_QMUL_MANIFEST` points to:

```text
manifest.json
```

## Input Files

The work directory contains:

```text
a.bin
b.bin
manifest.json
```

The first hardware command target is:

```text
float32 [N, 4] x [N, 4] -> [N, 4]
lane order: [real, i, j, k]
operation: Hamilton product
```

`a.bin` and `b.bin` are raw little-endian float32 row-major tensors.

## Output Files

The command must write:

```text
out.bin
metrics.json
```

`out.bin` must contain raw little-endian float32 row-major output with shape
`[N, 4]`.

## Required metrics.json Fields

`metrics.json` must contain:

```json
{
  "elapsed_s": 0.001,
  "device": "tenstorrent/<device>"
}
```

Required fields:

- `elapsed_s`: positive measured elapsed seconds for the benchmark iterations
- `device`: short device label used in StructuredBench reports

## Recommended metrics.json Fields

Recommended fields:

- `software_stack`
- `tt_metal_commit`
- `hardware_kind`
- `host`
- `notes`

Example:

```json
{
  "elapsed_s": 0.001,
  "device": "tenstorrent/wormhole",
  "software_stack": "tt-metal",
  "tt_metal_commit": "<commit>",
  "hardware_kind": "<device>",
  "host": "<host>",
  "notes": "Initial hardware validation sample; not a stable benchmark."
}
```

## Validation Command

Once a hardware command exists:

```bash
python scripts/rqm_tt_quickstart.py \
  --mode hardware \
  --command "<real Tenstorrent hardware qmul command>" \
  --items 128 \
  --iters 1 \
  --warmup 0 \
  --json-output reports/tt_hardware_qmul_quickstart.json \
  --markdown-output reports/tt_hardware_qmul_quickstart.md
```

The command must not be a TT-Lang simulator, tt-emule wrapper, or CPU reference
unless the report is labeled accordingly outside hardware mode.
