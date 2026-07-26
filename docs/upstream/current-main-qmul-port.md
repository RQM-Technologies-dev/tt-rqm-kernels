# Current-main qmul programming-example port

## Recorded revisions

- RQM source commit: `e9a8688e3766644bc0a0e6ef836b46465c04fde9`
- tt-metal base: `6d150126dcfa739669e3af63d60a5910a576d093`
- local branch: `codex/qmul-current-main-example`
- worktree: `tt-metal-qmul-current-main`

The port is local and uncommitted. It is not a Tenstorrent placement decision.

## Source mapping

| RQM candidate source | Upstream-shaped source |
|---|---|
| `src/qmul_multicore_candidate.cpp` | `contributed/qmul/qmul.cpp` |
| `kernels/qmul_multicore_reader.cpp` | `contributed/qmul/kernels/reader.cpp` |
| `kernels/qmul_multicore_compute.cpp` | `contributed/qmul/kernels/compute.cpp` |
| `kernels/qmul_sfpu.h` | `contributed/qmul/kernels/qmul_sfpu.h` |
| `kernels/qmul_multicore_writer.cpp` | `contributed/qmul/kernels/writer.cpp` |

All paths in the second column are relative to
`tt_metal/programming_examples/`. The port removes manifests, binary-file
protocols, timing/evidence fields, and RQM Python dependencies. Deterministic
inputs, Float64 whole-output validation, and a nonclaiming summary now live in
the C++ example.

## Current-main adaptations

Current `MeshDevice`, replicated `MeshBuffer`, `MeshWorkload`, queue,
`TensorAccessorArgs`, row-major `split_work_to_cores`, FP32 circular-buffer, and
`ComputeConfig` APIs remain compatible with the protected architecture.
Candidate-specific absolute kernel-path macros were replaced by the
programming-example `OVERRIDE_KERNEL_PREFIX`. Available runtime cores replace
the historical fixed Wormhole limit.

## Build and run

Narrow Linux build:

```bash
./build_metal.sh --build-programming-examples \
  --build-dir /tmp/build-qmul --cpm-source-cache /tmp/cpmcache \
  --configure-only --without-distributed --without-python-bindings \
  --disable-unity-builds --disable-profiler
cmake --build /tmp/build-qmul --target qmul -j12
```

Focused source check:

```bash
python3 tt_metal/programming_examples/contributed/qmul/tests/test_source_contract.py
```

Host-only executable checks:

```bash
/tmp/build-qmul/programming_examples/contributed/qmul --host-check --n 128
/tmp/build-qmul/programming_examples/contributed/qmul --host-check --n 4096
```

Hardware commands:

```bash
/tmp/build-qmul/programming_examples/contributed/qmul --n 128
/tmp/build-qmul/programming_examples/contributed/qmul --n 4096
```

## Validation and review boundary

The focused source contract check passes (5 tests), the current-main Linux
`qmul` target builds and links, and the built executable's host-only checks pass
for `N=128` and `N=4096` without device access.

On 2026-07-18, one configured N300 host ran each requested hardware case once
on device 0, with zero retries:

| N | Workers | Validated values | Failures | Nonfinite | Max absolute error | Max relative error | Result |
|---:|---:|---:|---:|---:|---:|---:|---|
| 128 | 1 | 512 | 0 | 0 | `4.842877e-08` | `1.548062e-06` | PASS |
| 4096 | 4 | 16,384 | 0 | 0 | `7.823110e-08` | `8.528494e-05` | PASS |

Both runs completed device closure cleanly. The host reported Wormhole N300,
TT-KMD 2.10.0, firmware bundle 19.6.0, and tt-metal base
`6d150126dcfa739669e3af63d60a5910a576d093`. These are single correctness runs,
not stable benchmark evidence or an acceleration claim.

Placement under `contributed`, the desirability of a custom SFPU helper in a
programming example, and any later experimental TT-NN surface remain maintainer
questions.

The historical qmul Claim Level 2 release applies only to its frozen candidate
and evidence. This current-main port inherits no benchmark or claim status.
