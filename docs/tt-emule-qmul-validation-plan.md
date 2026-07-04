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
- `tt-metal/build_emule` configure helper:
  `experimental/tt_emule_qmul/configure_build_emule.sh`
- `tt-metal/build_emule` build/install helper:
  `experimental/tt_emule_qmul/build_install_emule.sh`
- tracker issue:
  <https://github.com/RQM-Technologies-dev/tt-rqm-kernels/issues/8>
- installed TT-Metalium package under
  `/Users/home/Documents/tt-metal/build_emule/lib/cmake/tt-metalium`
- experimental candidate build against the installed `build_emule` prefix
- Docker wrapper for running the Linux candidate from the macOS
  StructuredBench `external-qmul` harness:
  `experimental/tt_metalium_qmul/run_candidate_docker.sh`
- emulation-labeled StructuredBench report:
  `reports/tt_emule_qmul_candidate.json` and
  `reports/tt_emule_qmul_candidate.md`

Not implemented:

- hardware report artifact

Current result:

```text
The Linux/tt-emule source-tree preflight passes, the local `tt-metal` checkout
is aligned to the `tt-emule` pin, required submodules are initialized,
`tt-metal/build_emule` configures with `TT_METAL_USE_EMULE=ON`, the build/install
step has produced usable installed TT-Metalium exports, and the experimental
TT-Metalium qmul source candidate now validates through `external-qmul` under
tt-emule.

This is emulation evidence only. It is not Tenstorrent hardware performance.
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

Updated local build state:

```text
tt-metal checkout: /Users/home/Documents/tt-metal @ dd2849b5bc6b7a5d38a9eafbeba31ef8d530f8d4
tt-metal submodule tt_metal/third_party/umd: initialized @ 12e28af324fe65223b8f10ff70fc060f4f184214
tt-metal submodule tt_metal/third_party/tracy: initialized @ 27221a69789f24d492135093e096b978b4ca3a68
build_emule configure: completed with TT_METAL_USE_EMULE=ON
build-prerequisite gate: source/toolchain/pin/submodule checks pass inside Ubuntu 24.04
TT-Metalium package install/export: completed under build_emule/lib/cmake/tt-metalium
candidate build: completed against the installed build_emule prefix
candidate validation: completed through external-qmul under tt-emule
emulation report: reports/tt_emule_qmul_candidate.json and reports/tt_emule_qmul_candidate.md
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

Current expected result in the prepared Docker/Linux environment is success once
`tt-metal/build_emule` has been configured and installed. A configured
build-tree config file alone is not enough unless the companion exported targets
are present.

Configure `build_emule`:

```bash
docker run --rm --platform linux/amd64 \
  -v /Users/home/Documents:/work \
  -w /work/tt-metal \
  -e TT_METAL_HOME=/work/tt-metal \
  -e TT_EMULE_HOME=/work/tt-emule \
  ubuntu:24.04 \
  bash /work/tt-rqm-kernels/experimental/tt_emule_qmul/configure_build_emule.sh
```

Build/install `build_emule`:

```bash
docker run --rm --platform linux/amd64 \
  -v /Users/home/Documents:/work \
  -w /work/tt-metal \
  -e TT_METAL_HOME=/work/tt-metal \
  -e TT_EMULE_HOME=/work/tt-emule \
  -e JOBS=2 \
  ubuntu:24.04 \
  bash -lc 'bash /work/tt-rqm-kernels/experimental/tt_emule_qmul/configure_build_emule.sh && \
            bash /work/tt-rqm-kernels/experimental/tt_emule_qmul/build_install_emule.sh'
```

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

Before rebuilding, confirm `tt-metal` remains aligned to the commit pinned by
`tt-emule/tt-metal-pin.txt` and required submodules remain initialized. The
candidate should be built against the installed/exported `build_emule` prefix so
the TT-Metalium package config can find its companion target files.

Candidate build command used for the current emulation evidence:

```bash
docker run --rm --platform linux/amd64 \
  -v /Users/home/Documents:/work \
  -w /work/tt-rqm-kernels \
  -e TT_METAL_HOME=/work/tt-metal \
  -e TT_EMULE_HOME=/work/tt-emule \
  ubuntu:24.04 \
  bash -lc 'apt-get update >/tmp/apt-update.log 2>&1 && \
            apt-get install -y --no-install-recommends ca-certificates cmake ninja-build g++ git python3 libopenmpi-dev openmpi-bin libhwloc-dev libnuma-dev >/tmp/apt-install.log 2>&1 && \
            python3 experimental/tt_metalium_qmul/build_candidate.py \
              --tt-metal-root /work/tt-metal \
              --cmake-prefix-path /work/tt-metal/build_emule \
              --build-dir /work/tt-rqm-kernels/experimental/tt_metalium_qmul/build_emule_candidate \
              --generator Ninja'
```

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

The current validation command is:

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

The generated report contains three 32-item `qmul` cases because the `qmul`
suite has three case slots and `--items 32` overrides each slot. This report is
intended to prove correctness and report shape under emulation, not stable
performance.

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
   JSON/Markdown reports labeled as emulation. This criterion has been met by
   `reports/tt_emule_qmul_candidate.json` and
   `reports/tt_emule_qmul_candidate.md`.
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
