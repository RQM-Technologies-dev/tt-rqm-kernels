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

This package intentionally does not include unverified TT-Metalium source code.
The local development environment does not currently provide a TT-Metalium
checkout, `ttnn`, `tt_metal`, or Tenstorrent hardware. The purpose of this
directory is to make the first real candidate contract explicit before hardware
code is added.

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

Use the candidate validation runner from the repository root:

```bash
python scripts/validate_qmul_candidate.py \
  --command "python scripts/qmul_external_reference.py" \
  --items 128 \
  --iters 1 \
  --warmup 0
```

For a future TT-Metalium binary, replace the command:

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

The future TT-Metalium implementation should start from the public programming
example pattern used by `tt-metal`: a host program plus data-movement and compute
kernels under a small example directory. The first version should prefer a
single-core row-major `[N, 4]` contract and correctness clarity over tiling or
peak performance.

The first hardware report must clearly state whether the result came from CPU,
TT-Lang simulation, emulation, or real Tenstorrent hardware.

## Non-Goals

- Do not add native quaternion hardware or a new datatype.
- Do not claim Tenstorrent endorsement.
- Do not claim hardware performance from CPU, simulator, or emulation output.
- Do not add upstream TT-Metalium code before placement guidance is clear.
