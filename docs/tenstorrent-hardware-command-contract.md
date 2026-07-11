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
  "schema": "tt-rqm-external-qmul-metrics.v2",
  "protocol": "tt-rqm-external-qmul.v1",
  "backend": "tt-metalium-qmul",
  "device": "tenstorrent/<device>",
  "dtype": "float32",
  "items": 128,
  "iterations": 1,
  "warmup": 0,
  "execution_kind": "hardware",
  "implementation_class": "<implementation>",
  "performance_eligible": false,
  "timings_s": {"setup": 0.1, "device": 0.001},
  "provenance": {
    "chip_type": "<chip>",
    "tt_metal_commit": "<commit>",
    "compiler_version": "<version>",
    "runtime_version": "<version>",
    "build_id": "<identity>",
    "timer_scope": "<exact measured scope>"
  }
}
```

Required fields:

- manifest-matching protocol, dtype, size, iteration, and warmup values
- non-empty backend, device, and implementation identity
- separate finite setup and device-execution timing
- complete chip, source, compiler, runtime, build, and timer-scope provenance
- `performance_eligible=false` for the scalar RISC-V correctness baseline

The harness independently measures end-to-end subprocess time, verifies timing
consistency, computes the candidate file hash, and records the repository
commit. A candidate-provided end-to-end number is not trusted as the primary
measurement.

## Benchmark Stages

- Stage A (`conformance`): exactly `N=128`, one repetition, one measured
  iteration, and `stable_benchmark=false`. This proves silicon compatibility.
- Stage B (`performance`): sizes 4,096, 65,536, and 262,144 with at least ten
  repetitions and reported median/p95 timing. It requires
  `performance_eligible=true`; the current scalar baseline cannot enter Stage B.

## Validation Command

Once a hardware command exists:

```bash
python scripts/rqm_tt_quickstart.py \
  --mode hardware \
  --benchmark-stage conformance \
  --command "<real Tenstorrent hardware qmul command>" \
  --items 128 \
  --iters 1 \
  --warmup 0 \
  --json-output reports/tt_hardware_qmul_quickstart.json \
  --markdown-output reports/tt_hardware_qmul_quickstart.md
```

The command must not be a TT-Lang simulator, tt-emule wrapper, or CPU reference
unless the report is labeled accordingly outside hardware mode.
