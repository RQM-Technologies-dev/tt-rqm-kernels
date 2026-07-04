# tt-emule qmul Validation Scaffold

This directory tracks the near-term emulation path for minimal `[N, 4]`
`qmul`.

It does not contain TT-Metalium kernel source. It only defines the preflight and
validation contract for a future candidate that builds with `tt-metal` using
`tt-emule`.

## Current Status

- CPU/PyTorch reference: implemented.
- TT-Lang simulator path: implemented and simulator-only.
- `external-qmul` candidate harness: implemented.
- TT-Metalium source for `qmul`: not implemented.
- tt-emule qmul candidate: not implemented.
- Local environment: this macOS checkout is not an x86-64 Linux tt-emule build
  environment.

## Preflight

Check whether the local machine has the expected tt-emule development layout:

```bash
python experimental/tt_emule_qmul/check_environment.py
```

Expected local result on unsupported platforms or without checkouts:

```text
tt-emule environment unavailable: ...
```

The checker expects:

```text
TT_METAL_HOME or TT_METALIUM_HOME -> local tt-metal checkout
TT_EMULE_HOME                     -> local tt-emule checkout
```

It also accepts explicit paths:

```bash
python experimental/tt_emule_qmul/check_environment.py \
  --tt-metal-root /path/to/tt-metal \
  --tt-emule-root /path/to/tt-emule
```

Passing this check only means the source tree layout is plausible. It does not
compile or run a kernel.

## Validation Contract

The future emulated candidate should be a standalone command that can be passed
to StructuredBench:

```bash
python scripts/validate_qmul_candidate.py \
  --command "/path/to/tt_emule_qmul_candidate" \
  --items 128 \
  --iters 1 \
  --warmup 0
```

StructuredBench provides:

```text
TT_RQM_EXTERNAL_QMUL_DIR
TT_RQM_EXTERNAL_QMUL_MANIFEST
a.bin
b.bin
manifest.json
```

The candidate writes:

```text
out.bin
metrics.json
```

`metrics.json` should label the device as emulation, for example:

```json
{
  "elapsed_s": 0.001,
  "device": "tt-emule/wormhole"
}
```

Do not label tt-emule output as hardware.

## Non-Goals

- No fake TT-Metalium source.
- No hardware-performance claims from emulation.
- No upstream `tt-metal` PR until placement guidance changes.
- No native quaternion datatype or hardware feature request.
