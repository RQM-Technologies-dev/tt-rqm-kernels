# TT-Metalium qmul Candidate Package

This directory is the staging area for a future minimal TT-Metalium `qmul`
candidate.

Current placement status:

- The Tenstorrent placement Discussion is open:
  https://github.com/tenstorrent/tt-metal/discussions/48871
- The active plan no longer waits on placement guidance.
- The candidate lives externally in `tt-rqm-kernels` by default and runs through
  the `external-qmul` StructuredBench harness.
- No upstream `tt-metal` PR should be opened from this directory unless
  actionable placement guidance arrives.

This package now includes an experimental TT-Metalium source candidate. It is a
minimal scalar RISC-V/data-movement implementation intended to prove the
`external-qmul` contract before any optimized compute-kernel work. It has now
been built against a local `tt-metal/build_emule` install and validated under
`tt-emule`. It has not been run on Tenstorrent hardware.

Current scaffold commands:

```text
check_environment.py  checks for a local tt-metal / TT-Metalium checkout
build_candidate.py    configures/builds the experimental CMake candidate
run_candidate.py      external-qmul wrapper for the built candidate binary
run_candidate_docker.sh
                      Docker wrapper that runs the Linux candidate from a macOS host
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

For the current Docker-backed tt-emule validation path from a macOS host, use:

```bash
python scripts/validate_qmul_candidate.py \
  --command "bash experimental/tt_metalium_qmul/run_candidate_docker.sh" \
  --execution-label emulation \
  --methodology-note "Experimental TT-Metalium qmul candidate run through tt-emule Docker wrapper; first validation sample, not a stable hardware benchmark." \
  --items 32 \
  --iters 1 \
  --warmup 0 \
  --json-output reports/tt_emule_qmul_candidate.json \
  --markdown-output reports/tt_emule_qmul_candidate.md
```

This produces an emulation-labeled StructuredBench report. It is not a hardware
performance result.

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

By default, the candidate binary is built under:

```text
experimental/tt_metalium_qmul/build_emule_candidate/tt_rqm_metalium_qmul_candidate
```

That path is intentionally aligned with `python scripts/rqm_tt_quickstart.py
--check` and with `run_candidate_docker.sh`. Use
`TT_RQM_METALIUM_QMUL_BINARY` only for a non-default local binary.

For `tt-emule`, first build/install `tt-metal` with `TT_METAL_USE_EMULE=ON`
following the `tt-emule` build guide, then pass the resulting `build_emule`
prefix.

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

Under tt-emule, raw L1 scratch pointer casts must use the direct
`reinterpret_cast<T*>(get_arg_val<uint32_t>(N))` source pattern so the JIT can
translate firmware L1 offsets to host pointers. The current kernel follows that
pattern for its scratch buffers.

Keep real candidate work in this external package and validate it through
`external-qmul`. Do not open an upstream `tt-metal` PR from this directory
unless actionable placement guidance arrives.

The first hardware report must clearly state whether the result came from CPU,
TT-Lang simulation, emulation, or real Tenstorrent hardware.

## Non-Goals

- Do not add native quaternion hardware or a new datatype.
- Do not claim Tenstorrent endorsement.
- Do not claim hardware performance from CPU, simulator, or emulation output.
- Do not add upstream TT-Metalium code before placement guidance is clear.
