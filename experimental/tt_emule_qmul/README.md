# tt-emule qmul Validation Scaffold

This directory tracks the near-term emulation path for minimal `[N, 4]`
`qmul`.

The TT-Metalium candidate source now lives in `experimental/tt_metalium_qmul`.
This directory defines the preflight and build-prerequisite checks for compiling
and running that candidate with `tt-metal` using `tt-emule`.

## Current Status

- CPU/PyTorch reference: implemented.
- TT-Lang simulator path: implemented and simulator-only.
- `external-qmul` candidate harness: implemented.
- TT-Metalium source for `qmul`: experimental scalar RISC-V candidate present.
- tt-emule qmul candidate: not implemented.
- Local x86-64 Linux Docker source-tree preflight: passes.
- Build prerequisites: not yet satisfied in the current container.

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

Check build prerequisites after the source-tree preflight:

```bash
python experimental/tt_emule_qmul/check_build_prereqs.py
```

This verifies the pinned `tt-metal` commit, required submodules, CMake/Ninja,
clang-20, and a built TT-Metalium CMake package such as `build_emule`.

## Validation Contract

The emulated candidate should be a standalone command that can be passed to
StructuredBench:

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
