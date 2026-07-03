# Structured [N, 4] Kernels

This tutorial shows how `tt-rqm-kernels` represents structured quaternion values
inside ordinary floating-point tensors, validates `qmul` on CPU/PyTorch, and
uses StructuredBench to prepare future Tenstorrent backend comparisons.

The intended reader is a Tenstorrent developer evaluating a small structured
numerical workload. This is not an official Tenstorrent tutorial, not a hardware
performance report, and not a request for native quaternion hardware.

## Why [N, 4] Structured Tensors Matter

Many accelerator examples focus on either scalar elementwise operators or large
matrix multiplication. Between those levels is a useful workload class:
structured values stored in dense tensor lanes.

`tt-rqm-kernels` starts with quaternion values:

```text
[N, 4] = [real, i, j, k]
```

The tensor is still an ordinary real-valued tensor. The structure comes from the
operator contract applied to the final dimension. That makes the data layout
compatible with tensor runtimes while preserving state that matters in robotics,
graphics, wireless, imaging, wave simulation, physical AI, scientific computing,
and signal processing.

The first benchmark target is `qmul`. It is compact enough to inspect by hand,
but structured enough to exercise fixed cross-lane dependencies, register reuse,
data movement, fusion opportunities, and arithmetic-intensity reporting.

## qmul Lane Equations

For two quaternion tensors:

```text
a = [a.w, a.x, a.y, a.z]
b = [b.w, b.x, b.y, b.z]
```

the Hamilton product is:

```text
out.w = a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z
out.x = a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y
out.y = a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x
out.z = a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w
```

Every output lane depends on all input lanes. For benchmark reporting,
StructuredBench counts each Hamilton product as 28 FLOPs: 16 multiplies plus
12 additions or subtractions.

## CPU/PyTorch Reference Run

Install the package for development:

```bash
python -m pip install -e ".[dev]"
```

Run a small CPU/PyTorch `qmul` benchmark:

```bash
python -m tt_rqm_kernels.structuredbench \
  --suite qmul \
  --items 128 \
  --iters 1 \
  --warmup 0
```

This emits a `structuredbench.v1` report with:

- latency and throughput
- max absolute, relative, and RMS numerical error
- scalar reference spot-check error
- estimated FLOPs/sec
- effective GB/sec
- arithmetic intensity
- checksum

The default backend is `torch`, so this is a CPU/PyTorch reference result unless
another backend is explicitly selected.

## Scalar Reference Checks

StructuredBench also compares a deterministic subset against scalar reference
functions in `tt_rqm_kernels.backends.scalar_reference`.

The scalar path matters because it is intentionally small and independent of the
vectorized PyTorch implementation. If a future candidate backend matches the
PyTorch reference but fails the scalar spot checks, the report should be treated
as a correctness failure.

The relevant scalar function for this tutorial is:

```text
tt_rqm_kernels.backends.scalar_reference.qmul_scalar
```

## StructuredBench Report Files

Generate sample JSON and Markdown reports:

```bash
python -m tt_rqm_kernels.structuredbench \
  --suite qmul \
  --items 128 \
  --iters 1 \
  --warmup 0 \
  --json-output reports/qmul_tutorial_cpu.json \
  --markdown-output reports/qmul_tutorial_cpu.md
```

Committed reports in this repository are sample reference outputs. They show the
report shape and comparison fields; they do not claim stable hardware
performance.

## Optional TT-Lang Simulator Smoke

If `tt-lang-sim` is installed, check availability:

```bash
python scripts/run_ttlang_qmul_smoke.py --check
```

Run the optional simulator path:

```bash
python -m tt_rqm_kernels.structuredbench \
  --backend tt-lang-sim \
  --suite qmul \
  --items 128 \
  --iters 1 \
  --warmup 0
```

The TT-Lang path is a functional simulator proof point. It validates kernel
logic and report shape. It is not Tenstorrent hardware execution and should not
be read as a hardware benchmark.

## external-qmul Candidate Harness

The `external-qmul` backend lets a standalone candidate executable plug into the
same validation and report path.

Run the CPU/PyTorch protocol reference command:

```bash
python scripts/validate_qmul_candidate.py \
  --command "python scripts/qmul_external_reference.py" \
  --items 128 \
  --iters 1 \
  --warmup 0
```

StructuredBench creates a temporary working directory and exposes it through:

```text
TT_RQM_EXTERNAL_QMUL_DIR
TT_RQM_EXTERNAL_QMUL_MANIFEST
```

The candidate command reads:

```text
a.bin
b.bin
manifest.json
```

and writes:

```text
out.bin
metrics.json
```

The binary tensors are raw little-endian float32 row-major values. `metrics.json`
must include a positive finite `elapsed_s` value and may include a `device` label
such as `cpu-reference`, `simulator`, `emulator`, or a real hardware identifier.

After the candidate exits, StructuredBench validates `out.bin` against the
CPU/PyTorch `qmul` reference and scalar spot checks, then emits the usual
`structuredbench.v1` report.

## Future TT-Metalium Plug-In Path

A future TT-Metalium executable should satisfy the same external contract:

```bash
python scripts/validate_qmul_candidate.py \
  --command "/path/to/tt_metalium_qmul_candidate" \
  --items 4096 \
  --iters 10 \
  --warmup 2 \
  --json-output reports/tt_metalium_qmul_candidate.json \
  --markdown-output reports/tt_metalium_qmul_candidate.md
```

The staging area for this work is:

```text
experimental/tt_metalium_qmul/
```

Until Tenstorrent maintainers give placement guidance, the first candidate
should remain external to `tt-metal`. It should not be presented as an official
Tenstorrent example, a native quaternion feature, or a hardware performance
claim.

## Non-Goals

- No native quaternion datatype.
- No new silicon feature.
- No defense-first framing.
- No claim that simulator output is hardware output.
- No claim of Tenstorrent endorsement.
- No speculative physics claims.

The technical goal is narrower: make a compact structured-kernel benchmark that
can compare CPU/PyTorch references, scalar checks, simulator paths, and future
Tenstorrent backend candidates using the same report shape.
