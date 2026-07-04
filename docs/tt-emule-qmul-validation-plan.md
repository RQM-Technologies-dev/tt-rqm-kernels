# tt-emule qmul Validation Plan

## Purpose

This plan defines the next lower-stack validation step for StructuredBench
`qmul`: a future TT-Metalium candidate compiled and run through `tt-emule`.

The goal is executable emulation evidence, not hardware performance. `tt-emule`
is useful here because it can run `tt-metal` host/kernel code on an x86-64 Linux
machine without Tenstorrent hardware. Reports from this path must be labeled as
emulation.

## Current Status

Implemented:

- CPU/PyTorch `qmul` reference
- scalar correctness spot checks
- TT-Lang functional simulator `qmul`
- `external-qmul` candidate harness
- TT-Metalium candidate scaffold
- tt-emule environment preflight:
  `experimental/tt_emule_qmul/check_environment.py`
- x86-64 Linux Docker preflight with sibling `tt-metal` and `tt-emule`
  checkouts
- experimental TT-Metalium scalar RISC-V `qmul` candidate source:
  `experimental/tt_metalium_qmul/src/qmul_candidate.cpp`
- build-prerequisite checker:
  `experimental/tt_emule_qmul/check_build_prereqs.py`
- tracker issue:
  <https://github.com/RQM-Technologies-dev/tt-rqm-kernels/issues/8>

Not implemented:

- tt-emule build of a `qmul` candidate
- emulation report artifact
- hardware report artifact

Current blocker:

```text
The Linux/tt-emule source-tree preflight now passes and an experimental
TT-Metalium qmul source candidate exists. The remaining blocker is a real
tt-metal/tt-emule build environment: the local tt-metal checkout is not on the
tt-emule pinned commit, required submodules are not initialized, clang-20 /
git / CMake / Ninja are not present in the python:3.12-slim container, and no
build_emule TT-Metalium CMake package exists yet.
```

Verified preflight environment:

```text
Docker Desktop daemon: linux x86_64, server 29.2.1
Container image: python:3.12-slim with --platform linux/amd64
Mounted workspace: /Users/home/Documents -> /work
tt-metal checkout: /Users/home/Documents/tt-metal @ fd810266
tt-emule checkout: /Users/home/Documents/tt-emule @ abdc348
tt-rqm-kernels checkout: /Users/home/Documents/tt-rqm-kernels
tt-emule pinned tt-metal commit: dd2849b5bc6b7a5d38a9eafbeba31ef8d530f8d4
```

## Expected Checkout Layout

Use current upstream documentation as the setup source of truth. The intended
local layout is:

```text
$ROOT/
  tt-metal/
  tt-emule/
  tt-rqm-kernels/
```

Environment variables:

```bash
export TT_METAL_HOME=$ROOT/tt-metal
export TT_EMULE_HOME=$ROOT/tt-emule
```

Preflight:

```bash
docker run --rm --platform linux/amd64 \
  -v /Users/home/Documents:/work \
  -w /work/tt-rqm-kernels \
  -e TT_METAL_HOME=/work/tt-metal \
  -e TT_EMULE_HOME=/work/tt-emule \
  python:3.12-slim \
  python experimental/tt_emule_qmul/check_environment.py
```

Current result:

```text
tt-metal root detected: /work/tt-metal
tt-emule root detected: /work/tt-emule
tt-emule qmul preflight passed. This does not run a kernel.
```

This check only validates platform and source-tree layout. It does not prove the
candidate builds, runs, or produces correct output.

Build-prerequisite check:

```bash
docker run --rm --platform linux/amd64 \
  -v /Users/home/Documents:/work \
  -w /work/tt-rqm-kernels \
  -e TT_METAL_HOME=/work/tt-metal \
  -e TT_EMULE_HOME=/work/tt-emule \
  python:3.12-slim \
  python experimental/tt_emule_qmul/check_build_prereqs.py
```

Current expected result is exit code 2 with concrete setup blockers. This is
not a candidate failure; it means the Linux container still lacks the toolchain,
the pinned `tt-metal` checkout, initialized submodules, and a built
TT-Metalium package.

## Build Direction

The candidate source now follows the `add_2_integers_in_riscv` public
programming example pattern: a host program plus a RISC-V data-movement kernel.
It maps one quaternion to one 16-byte page and implements the Hamilton product
lane equations directly for correctness validation. The build should follow the
`tt-emule` integration pattern from the `tt-emule` README:

```bash
cmake -S "$TT_METAL_HOME" -B "$TT_METAL_HOME/build_emule" \
  -DTT_METAL_USE_EMULE=ON \
  -DTT_EMULE_PATH="$TT_EMULE_HOME"
cmake --build "$TT_METAL_HOME/build_emule" -j"$(nproc)"
```

Before building, align `tt-metal` to the commit pinned by
`tt-emule/tt-metal-pin.txt`, initialize required submodules, and use an image or
host environment with clang-20, CMake, and Ninja. Exact flags and target names
should be confirmed in the Linux environment before publishing any build claim.

## external-qmul Mapping

The emulated candidate should satisfy the existing `external-qmul` protocol:

```text
[N, 4] x [N, 4] -> [N, 4]
lane order: [real, i, j, k]
dtype: float32
```

StructuredBench provides a temporary working directory containing:

```text
a.bin
b.bin
manifest.json
```

The candidate must write:

```text
out.bin
metrics.json
```

`metrics.json` must include a positive finite `elapsed_s` and a device label
that clearly says emulation:

```json
{
  "elapsed_s": 0.001,
  "device": "tt-emule/wormhole"
}
```

The validation command should remain:

```bash
python scripts/validate_qmul_candidate.py \
  --command "/path/to/tt_emule_qmul_candidate" \
  --items 128 \
  --iters 1 \
  --warmup 0 \
  --json-output reports/tt_emule_qmul_candidate.json \
  --markdown-output reports/tt_emule_qmul_candidate.md
```

## Required Report Labels

Every tt-emule report must state:

```text
backend: external-qmul
device: tt-emule/<target>
execution_label: emulation
stable_benchmark: false
```

Reports must include the normal `structuredbench.v1` fields:

- latency
- throughput
- numerical errors
- scalar reference error
- estimated FLOPs/sec
- effective GB/sec
- arithmetic intensity
- checksum

## Acceptance Criteria

The tt-emule milestone is ready to close when either:

1. A `qmul` candidate runs through `external-qmul` under tt-emule and produces
   JSON/Markdown reports labeled as emulation.
2. The blocker is documented with exact failing commands, platform, checkout
   SHAs, and logs precise enough for a Tenstorrent maintainer or collaborator to
   reproduce.

## Non-Goals

- Do not add placeholder TT-Metalium source that does not build against a real
  SDK checkout.
- Do not claim tt-emule output is hardware performance.
- Do not open an upstream Tenstorrent PR from this milestone.
- Do not request native quaternion hardware, new silicon features, or official
  Tenstorrent endorsement.

## References

- tt-emule: <https://github.com/tenstorrent/tt-emule>
- TT-Metalium docs:
  <https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/>
- StructuredBench spec: `docs/structuredbench-spec.md`
- qmul design: `docs/tt-metalium-qmul-design.md`
- external candidate scaffold: `experimental/tt_metalium_qmul/README.md`
