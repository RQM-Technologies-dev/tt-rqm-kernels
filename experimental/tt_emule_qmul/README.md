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
- tt-emule qmul candidate: experimental emulation report present.
- Local x86-64 Linux Docker source-tree preflight: passes.
- Build prerequisites: satisfied in the prepared Ubuntu 24.04 Docker/Linux
  environment after `tt-metal/build_emule` install.

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
clang-20, and a usable built/installed TT-Metalium CMake package. A configured
build-tree config file is not enough unless the companion exported targets are
present.

Configure `tt-metal/build_emule` from the host with:

```bash
docker run --rm --platform linux/amd64 \
  -v /Users/home/Documents:/work \
  -w /work/tt-metal \
  -e TT_METAL_HOME=/work/tt-metal \
  -e TT_EMULE_HOME=/work/tt-emule \
  ubuntu:24.04 \
  bash /work/tt-rqm-kernels/experimental/tt_emule_qmul/configure_build_emule.sh
```

This installs build dependencies inside the disposable container and writes the
CMake build directory to the mounted `tt-metal/build_emule` path.

Build/install `tt-metal/build_emule` after configuration with:

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

The current prepared environment has completed the `build_emule` install and
provides the installed TT-Metalium package exports under
`tt-metal/build_emule/lib/cmake/tt-metalium`.

Build the experimental candidate against that installed prefix with:

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

Current Docker-backed validation command from the macOS host:

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

The generated report is emulation-only:

```text
backend=external-qmul
device=tt-emule/tt-metalium-riscv-qmul-candidate
execution_label=emulation
stable_benchmark=false
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
