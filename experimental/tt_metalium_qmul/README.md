# TT-Metalium qmul Candidate Package

This directory is the staging area for a future minimal TT-Metalium `qmul`
candidate.

Current placement status:

- The Tenstorrent placement Discussion is open:
  https://github.com/tenstorrent/tt-metal/discussions/48871
- No maintainer placement answer has been received yet.
- Until placement guidance is received, the candidate should live externally in
  `tt-rqm-kernels` and run through the `external-qmul` StructuredBench harness.
- No upstream `tt-metal` PR should be opened from this directory without
  maintainer guidance.

This package now includes an experimental TT-Metalium source candidate. It is a
minimal scalar RISC-V/data-movement implementation intended to prove the
`external-qmul` contract before any optimized compute-kernel work. It has not
yet been built under `tt-emule` or run on Tenstorrent hardware.

Current scaffold commands:

```text
check_environment.py  checks for a local tt-metal / TT-Metalium checkout
build_candidate.py    configures/builds the experimental CMake candidate
run_candidate.py      external-qmul wrapper for the built candidate binary
validate_candidate.py wrapper around scripts/validate_qmul_candidate.py
```

The run wrapper is expected to fail until the candidate binary is built in a
real TT-Metalium environment. That failure is intentional: it prevents CPU,
simulator, or empty output from being mistaken for Tenstorrent hardware results.

## Candidate Contract

The first candidate command must implement:

```text
[N, 4] x [N, 4] -> [N, 4]
```

with lane order:

```text
[real, i, j, k]
```

StructuredBench will provide a temporary work directory through:

```text
TT_RQM_EXTERNAL_QMUL_DIR
TT_RQM_EXTERNAL_QMUL_MANIFEST
```

The candidate command must read:

```text
a.bin
b.bin
manifest.json
```

and write:

```text
out.bin
metrics.json
```

Binary tensors are raw little-endian float32 row-major values. For `N` items,
each tensor contains `4 * N` float32 values.

`metrics.json` must include:

```json
{
  "elapsed_s": 0.001,
  "device": "candidate-device-label"
}
```

`elapsed_s` must be a positive finite wall-clock duration for the requested
iteration loop. `device` should distinguish CPU reference, simulator, emulation,
and real Tenstorrent hardware.

## Validation Command

Use the known-good CPU/PyTorch protocol reference from the repository root:

```bash
python scripts/validate_qmul_candidate.py \
  --command "python scripts/qmul_external_reference.py" \
  --items 128 \
  --iters 1 \
  --warmup 0
```

Use the local wrapper to validate the same protocol reference through this
candidate package:

```bash
python experimental/tt_metalium_qmul/validate_candidate.py \
  --candidate-command "python scripts/qmul_external_reference.py" \
  --items 128 \
  --iters 1 \
  --warmup 0
```

The candidate wrapper can be invoked directly to verify failure behavior:

```bash
python experimental/tt_metalium_qmul/run_candidate.py
```

Without `TT_RQM_EXTERNAL_QMUL_DIR` and `TT_RQM_EXTERNAL_QMUL_MANIFEST`, it exits
with an external-harness environment error. When run through StructuredBench
without a TT-Metalium SDK or built candidate binary, it exits clearly and writes
no `out.bin` or `metrics.json`.

For a future TT-Metalium binary, replace the command:

```bash
python experimental/tt_metalium_qmul/validate_candidate.py \
  --candidate-command "/path/to/tt_metalium_qmul_candidate" \
  --items 128 \
  --iters 1 \
  --warmup 0
```

For a larger report once the candidate is real:

```bash
python scripts/validate_qmul_candidate.py \
  --command "/path/to/tt_metalium_qmul_candidate" \
  --items 4096 \
  --iters 10 \
  --warmup 2 \
  --json-output reports/tt_metalium_qmul_candidate.json \
  --markdown-output reports/tt_metalium_qmul_candidate.md
```

StructuredBench validates output against the CPU/PyTorch reference and scalar
spot checks before reporting:

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

## Implementation Notes

Check for a local SDK checkout before attempting real candidate development:

```bash
python experimental/tt_metalium_qmul/check_environment.py
```

The build command checks the SDK environment and then requires a built or
installed TT-Metalium CMake package:

```bash
python experimental/tt_metalium_qmul/build_candidate.py \
  --tt-metal-root /path/to/tt-metal \
  --cmake-prefix-path /path/to/tt-metal/build_emule
```

For `tt-emule`, first build `tt-metal` with `TT_METAL_USE_EMULE=ON` following
the `tt-emule` build guide, then pass the resulting `build_emule` prefix.

The checker looks for `TT_METAL_HOME` or `TT_METALIUM_HOME`, or accepts:

```bash
python experimental/tt_metalium_qmul/check_environment.py \
  --tt-metal-root /path/to/tt-metal
```

The current source starts from the public `add_2_integers_in_riscv` programming
example pattern used by `tt-metal`: a host program plus a data-movement RISC-V
kernel. It maps one quaternion to one 16-byte page and implements the Hamilton
product lane equations directly. This is a correctness candidate, not an
optimized kernel design.

Until maintainers answer the placement question, keep real candidate work in
this external package and validate it through `external-qmul`. Do not open an
upstream `tt-metal` PR from this directory without maintainer guidance.

The first hardware report must clearly state whether the result came from CPU,
TT-Lang simulation, emulation, or real Tenstorrent hardware.

## Non-Goals

- Do not add native quaternion hardware or a new datatype.
- Do not claim Tenstorrent endorsement.
- Do not claim hardware performance from CPU, simulator, or emulation output.
- Do not add upstream TT-Metalium code before placement guidance is clear.
