# Tenstorrent Engineer Copy/Paste Packet

This packet is for a Tenstorrent engineer or maintainer with access to a real
Tenstorrent hardware environment. It is intended to produce one
hardware-labeled StructuredBench `qmul` report from the current experimental
TT-Metalium scalar RISC-V candidate.

This is not an upstream `tt-metal` PR, not a request for native quaternion
hardware, and not a stable benchmark claim.

## Goal

Produce:

```text
reports/tt_hardware_qmul_quickstart.json
reports/tt_hardware_qmul_quickstart.md
```

for:

```text
float32 [N, 4] x [N, 4] -> [N, 4]
operation: Hamilton product qmul
lane order: [real, i, j, k]
```

The first run should use:

```text
N=128
iters=1
warmup=0
execution_label=hardware
stable_benchmark=false
```

## Environment

Use either:

- a Tenstorrent Console managed VSCode/browser instance
- a Tenstorrent baremetal SSH shell
- an internal Tenstorrent hardware environment

Required local tools in that environment:

- Python 3.10 or newer
- CMake and a C++ compiler usable with the local TT-Metalium stack
- a `tt-metal` / TT-Metalium checkout or install with CMake package exports
- real Tenstorrent hardware access

Set these paths for the local environment:

```bash
export TT_METAL_HOME=/path/to/tt-metal
export TT_METALIUM_PREFIX=/path/to/tt-metal/build
```

If the environment uses a different installed TT-Metalium prefix, set
`TT_METALIUM_PREFIX` to that install prefix.

## Clone And Install

```bash
git clone https://github.com/RQM-Technologies-dev/tt-rqm-kernels.git
cd tt-rqm-kernels
python -m pip install -e ".[dev]"
```

Record the repo commit:

```bash
git rev-parse HEAD
```

## CPU Reference Checks

```bash
python -m pytest
python -m tt_rqm_kernels.structuredbench \
  --suite smoke \
  --items 128 \
  --iters 1 \
  --warmup 0
```

Validate the `external-qmul` protocol with the CPU reference command:

```bash
python scripts/validate_qmul_candidate.py \
  --command "python scripts/qmul_external_reference.py" \
  --items 128 \
  --iters 1 \
  --warmup 0
```

## Build The Experimental Candidate

The candidate source lives in:

```text
experimental/tt_metalium_qmul/
```

Build it against the local TT-Metalium package:

```bash
export TT_RQM_CANDIDATE_BUILD="$PWD/experimental/tt_metalium_qmul/build_hardware_candidate"

python experimental/tt_metalium_qmul/build_candidate.py \
  --tt-metal-root "$TT_METAL_HOME" \
  --cmake-prefix-path "$TT_METALIUM_PREFIX" \
  --build-dir "$TT_RQM_CANDIDATE_BUILD" \
  --generator Ninja
```

The expected candidate executable is:

```text
experimental/tt_metalium_qmul/build_hardware_candidate/tt_rqm_metalium_qmul_candidate
```

If the build fails because the local TT-Metalium export layout differs, return
the build log and the `tt-metal` commit instead of fabricating a report.

## Run Hardware Validation

Run the built candidate through StructuredBench:

```bash
python scripts/validate_qmul_candidate.py \
  --command "$TT_RQM_CANDIDATE_BUILD/tt_rqm_metalium_qmul_candidate" \
  --execution-label hardware \
  --methodology-note "Initial Tenstorrent hardware validation sample for experimental TT-Metalium scalar RISC-V qmul candidate; not a stable benchmark." \
  --items 128 \
  --iters 1 \
  --warmup 0 \
  --json-output reports/tt_hardware_qmul_quickstart.json \
  --markdown-output reports/tt_hardware_qmul_quickstart.md
```

The command must read the `external-qmul` inputs provided by StructuredBench and
write `out.bin` and `metrics.json`. StructuredBench validates output against
CPU/PyTorch and scalar references before writing the report artifacts.

## Return

Please return:

- `reports/tt_hardware_qmul_quickstart.json`
- `reports/tt_hardware_qmul_quickstart.md`
- environment notes

Environment notes should include:

- hardware kind
- host or Console environment label
- `tt-metal` / TT-Metalium commit or package version
- Python version
- exact command used for the candidate
- any local build or runtime changes needed

## Label Rules

- Use `execution_label=hardware` only if the candidate ran on real Tenstorrent
  hardware.
- Keep `stable_benchmark=false` for this first sample.
- Do not return CPU, TT-Lang simulator, or tt-emule output as a hardware
  report.
- Do not claim stable hardware performance from a first validation sample.
