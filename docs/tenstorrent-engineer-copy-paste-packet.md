# Tenstorrent Engineer Copy/Paste Packet

This packet is for a Tenstorrent engineer or maintainer with access to a real
Tenstorrent hardware environment. It is intended to produce one
hardware-labeled StructuredBench `qmul` report from the current experimental
TT-Metalium scalar RISC-V candidate.

This is not an upstream `tt-metal` PR, not a request for native quaternion
hardware, not placement guidance work, and not a stable benchmark claim.

## Goal

Return these artifacts:

```text
reports/tt_hardware_qmul_quickstart.json
reports/tt_hardware_qmul_quickstart.md
reports/tt_hardware_qmul_environment.txt
```

for:

```text
float32 [N, 4] x [N, 4] -> [N, 4]
operation: Hamilton product qmul
lane order: [real, i, j, k]
N=128
iters=1
warmup=0
execution_label=hardware
benchmark_stage=conformance
stable_benchmark=false
```

Use `execution_label=hardware` only if the candidate ran on real Tenstorrent
hardware. First samples should keep `stable_benchmark=false`.

## Environment

Use one of:

- a Tenstorrent Console managed VSCode/browser instance
- a Tenstorrent baremetal SSH shell
- an internal Tenstorrent hardware environment

Required tools:

- Python 3.10 or newer
- CMake, Ninja, and a C++ compiler usable with the local TT-Metalium stack
- a `tt-metal` / TT-Metalium checkout or install with CMake package exports
- real Tenstorrent hardware access

Set these paths for the local environment before running the sequence:

```bash
export TT_METAL_HOME=/path/to/tt-metal
export TT_METALIUM_PREFIX=/path/to/tt-metal/build
```

If the environment uses a different installed TT-Metalium prefix, set
`TT_METALIUM_PREFIX` to that install prefix.

## Copy/Paste Sequence

```bash
set -euo pipefail

git clone https://github.com/RQM-Technologies-dev/tt-rqm-kernels.git
cd tt-rqm-kernels
python -m pip install -e ".[dev]"
mkdir -p reports

python scripts/rqm_tt_quickstart.py --check
python -m pytest tests/test_tenstorrent_adapter.py tests/test_repo_status.py -q
python -m tt_rqm_kernels.structuredbench \
  --suite smoke \
  --items 128 \
  --iters 1 \
  --warmup 0

python scripts/validate_qmul_candidate.py \
  --command "python scripts/qmul_external_reference.py" \
  --items 128 \
  --iters 1 \
  --warmup 0

export TT_RQM_CANDIDATE_BUILD="$PWD/experimental/tt_metalium_qmul/build_hardware_candidate"

python experimental/tt_metalium_qmul/build_candidate.py \
  --tt-metal-root "$TT_METAL_HOME" \
  --cmake-prefix-path "$TT_METALIUM_PREFIX" \
  --build-dir "$TT_RQM_CANDIDATE_BUILD" \
  --generator Ninja

export TT_RQM_CHIP_TYPE="${TT_RQM_CHIP_TYPE:?set TT_RQM_CHIP_TYPE to the real chip type}"
export TT_RQM_TT_METAL_COMMIT="$(git -C "$TT_METAL_HOME" rev-parse HEAD)"
export TT_RQM_COMPILER_VERSION="$(c++ --version | head -n 1)"
export TT_RQM_RUNTIME_VERSION="$TT_RQM_TT_METAL_COMMIT"
export TT_RQM_BUILD_ID="$(sha256sum "$TT_RQM_CANDIDATE_BUILD/tt_rqm_metalium_qmul_candidate" | cut -d' ' -f1)"

python scripts/validate_qmul_candidate.py \
  --command "$TT_RQM_CANDIDATE_BUILD/tt_rqm_metalium_qmul_candidate" \
  --execution-label hardware \
  --benchmark-stage conformance \
  --methodology-note "Initial Tenstorrent hardware validation sample for experimental TT-Metalium scalar RISC-V qmul candidate; not a stable benchmark." \
  --items 128 \
  --iters 1 \
  --warmup 0 \
  --json-output reports/tt_hardware_qmul_quickstart.json \
  --markdown-output reports/tt_hardware_qmul_quickstart.md

{
  echo "repo_commit=$(git rev-parse HEAD)"
  echo "python=$(python --version 2>&1)"
  echo "tt_metal_home=$TT_METAL_HOME"
  echo "tt_metalium_prefix=$TT_METALIUM_PREFIX"
  if [ -d "$TT_METAL_HOME/.git" ]; then
    echo "tt_metal_commit=$(git -C "$TT_METAL_HOME" rev-parse HEAD)"
  fi
  echo "hardware_kind=<fill in>"
  echo "host_or_console_label=<fill in>"
  echo "candidate_command=$TT_RQM_CANDIDATE_BUILD/tt_rqm_metalium_qmul_candidate"
  echo "local_changes_or_runtime_notes=<fill in or none>"
} > reports/tt_hardware_qmul_environment.txt
```

The candidate command must implement the
[Tenstorrent hardware command contract](tenstorrent-hardware-command-contract.md):
StructuredBench provides `a.bin`, `b.bin`, and `manifest.json`; the candidate
writes `out.bin` and strict metrics-v2 metadata. StructuredBench validates every
output value against an independent float64 golden result from the exact float32
inputs before writing the report artifacts.

If the build fails because the local TT-Metalium export layout differs, return
the build log and the `tt-metal` commit instead of fabricating a report.

## Return

Please return:

- `reports/tt_hardware_qmul_quickstart.json`
- `reports/tt_hardware_qmul_quickstart.md`
- `reports/tt_hardware_qmul_environment.txt`
- any build/runtime log if the sequence fails

## Label Rules

- Use `execution_label=hardware` only for real Tenstorrent hardware execution.
- Keep `stable_benchmark=false` for the first sample.
- Do not return CPU, TT-Lang simulator, Docker, or tt-emule output as a hardware
  report.
- Do not claim stable hardware performance from a first validation sample.
