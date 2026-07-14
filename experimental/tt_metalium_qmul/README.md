# TT-Metalium qmul Candidates

This package contains the external TT-Metalium `qmul` candidates used by the
repository's `external-qmul` StructuredBench harness. The wire operation is:

```text
float32 [N, 4] x [N, 4] -> [N, 4]
lane order: [real, i, j, k]
operation: Hamilton product
```

The scalar RISC-V/data-movement candidate has passed the Stage A N300 silicon
conformance gate. That result is recorded in the
[JSON report](../../reports/tt_hardware_qmul_quickstart.json),
[Markdown report](../../reports/tt_hardware_qmul_quickstart.md), and
[environment record](../../reports/tt_hardware_qmul_environment.txt).
The hardware binary ran from clean source commit
[`f73221f`](https://github.com/RQM-Technologies-dev/tt-rqm-kernels/commit/f73221f014c2ea0c1ad9b44fbfd44c5492859943);
the resulting evidence and documentation were committed afterward in
[`efa529a`](https://github.com/RQM-Technologies-dev/tt-rqm-kernels/commit/efa529a59bc709fccb58d6134dedff3297f8fdaa).
This provenance distinction is intentional: the report and environment record
agree on the source commit that produced the executed binary.

The scalar candidate is a correctness baseline only. It has
`performance_eligible=false` and is permanently ineligible for Stage B. Never
run it with Stage B sizes, iterations, warmups, or repetitions. Stage B uses a
separate multicore compute/SFPU candidate and separate protected artifacts.

The Tenstorrent placement Discussion remains open at
<https://github.com/tenstorrent/tt-metal/discussions/48871>. Candidate work
stays external to `tt-metal` unless actionable placement guidance arrives.

## Package Commands

```text
check_environment.py  checks for a local tt-metal / TT-Metalium checkout
build_candidate.py    configures and builds a selected experimental candidate
run_candidate.py      external-qmul wrapper for the built candidate binary
run_candidate_docker.sh
                      Docker wrapper for the scalar tt-emule path on macOS
validate_candidate.py wrapper around scripts/validate_qmul_candidate.py
```

The wrapper fails closed when the SDK, candidate binary, or harness environment
is missing. It must not produce empty output or let CPU, simulator, or emulation
execution appear as Tenstorrent hardware evidence.

## External-qmul Wire Contract

StructuredBench provides a temporary work directory through:

```text
TT_RQM_EXTERNAL_QMUL_DIR
TT_RQM_EXTERNAL_QMUL_MANIFEST
```

The candidate reads `a.bin`, `b.bin`, and `manifest.json`, then writes `out.bin`
and `metrics.json` in that directory. Binary tensors are raw little-endian
float32 row-major values. Each tensor contains `4 * N` values.

`metrics.json` must satisfy the complete
`tt-rqm-external-qmul-metrics.v2` contract:

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
  "timings_s": {
    "setup": 0.1,
    "device": 0.001
  },
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

The protocol, dtype, items, iterations, warmup, and execution kind must match
the manifest and invocation. Backend, device, and implementation class must be
non-empty. Setup may be zero; device time must be positive and finite. Hardware
runs require every provenance field shown above. The harness independently
records end-to-end time, the repository commit, and the observed candidate
binary hash; any candidate-provided hash must match that observed binary.

See the canonical
[hardware command contract](../../docs/tenstorrent-hardware-command-contract.md)
for the enforcement rules.

## Validation

Validate the protocol itself with the repository's CPU/PyTorch reference:

```bash
python scripts/validate_qmul_candidate.py \
  --command "python scripts/qmul_external_reference.py" \
  --items 128 \
  --iters 1 \
  --warmup 0
```

Validate the built scalar candidate only with the Stage A conformance shape:

```bash
python experimental/tt_metalium_qmul/validate_candidate.py \
  --candidate-command "/path/to/tt_rqm_metalium_qmul_candidate" \
  --items 128 \
  --iters 1 \
  --warmup 0
```

For the Docker-backed tt-emule path from a macOS host:

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

Emulation output is not hardware performance evidence. The official Stage A
hardware invocation is maintained in the
[execution runbook](../../docs/tenstorrent-execution-runbook.md). Stage B must
use the separate multicore candidate and its stage-specific commands.

StructuredBench performs whole-output validation against an independent
float64 Hamilton-product golden result using `atol=1e-4` and `rtol=1e-4`. It
also rejects non-finite values, reports failing and validated value counts,
checks scalar reference values, validates metrics-v2 and provenance, and
records raw setup, device, and end-to-end timing samples.

## Build and Implementation Notes

Check the SDK before building:

```bash
python experimental/tt_metalium_qmul/check_environment.py \
  --tt-metal-root /path/to/tt-metal
```

Build against a completed TT-Metalium CMake package:

```bash
python experimental/tt_metalium_qmul/build_candidate.py \
  --tt-metal-root /path/to/tt-metal \
  --cmake-prefix-path /path/to/tt-metal/build_emule
```

The scalar source follows TT-Metalium's public
`add_2_integers_in_riscv` host/data-movement pattern. It maps one quaternion to
one 16-byte page and evaluates the Hamilton equations on the scalar
data-movement RISC-V. That architecture is retained only as the immutable Stage
A baseline.

Under tt-emule, raw L1 scratch pointer casts use the direct
`reinterpret_cast<T*>(get_arg_val<uint32_t>(N))` form so the JIT can translate
firmware L1 offsets to host pointers.

## Non-Goals

- Do not add native quaternion hardware or a new datatype.
- Do not claim Tenstorrent endorsement.
- Do not claim hardware performance from CPU, simulator, or emulation output.
- Do not use the scalar Stage A candidate for Stage B.
- Do not open an upstream `tt-metal` PR without actionable placement guidance.
