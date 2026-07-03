# TT-Metalium qmul Design

This document defines the minimal future TT-Metalium `qmul` example for
`tt-rqm-kernels`. It is a design target, not an implementation.

The goal is to make the first hardware-facing kernel small, testable, and easy
for Tenstorrent maintainers to place correctly before any TT-Metalium code is
written.

## Purpose

The first TT-Metalium target is quaternion multiply over structured `[N, 4]`
floating-point tensors.

This workload is useful because it is compact but not scalar-trivial:

- every output lane depends on every input lane
- the multiply/add/sign pattern is fixed and easy to audit
- correctness can be checked against existing CPU/PyTorch and scalar references
- benchmark output can reuse the existing StructuredBench report shape

The first version should prioritize clarity, correctness, and reproducible
benchmarking over peak performance.

## Operator Contract

Minimal transform:

```text
[N, 4] x [N, 4] -> [N, 4]
```

Inputs:

```text
a: [N, 4]
b: [N, 4]
```

Output:

```text
out: [N, 4]
```

Lane order:

```text
[real, i, j, k]
```

Initial constraints:

- `N` is the number of quaternion items.
- Final dimension size is exactly `4`.
- Inputs and output are ordinary floating-point tensors.
- The first TT-Metalium example should use a simple flattened `[N, 4]`
  contract before considering richer broadcasting or batched layouts.

## Hamilton Product

The TT-Metalium result must match the reference `qmul` contract:

```text
out.w = a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z
out.x = a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y
out.y = a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x
out.z = a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w
```

The expected operation count for reporting is 28 FLOPs per Hamilton product:
16 multiplies plus 12 additions or subtractions.

## Reference Validation

The validation source of truth is the existing CPU/PyTorch implementation:

```text
tt_rqm_kernels.quaternion_ops.qmul
```

The independent scalar spot-check source is:

```text
tt_rqm_kernels.backends.scalar_reference.qmul_scalar
```

Recommended validation path:

1. Generate deterministic `[N, 4]` input tensors.
2. Run CPU/PyTorch `qmul` as the golden reference.
3. Run the TT-Metalium `qmul` candidate.
4. Compare elementwise max absolute error, max relative error, and RMS error.
5. Run scalar spot checks on a small deterministic subset.
6. Include identity and basis multiplication cases in unit-level validation.
7. Record dtype, input size, seed, and backend identity in the report.

The TT-Lang simulator report is useful as a pre-hardware logic check, but it is
not a substitute for TT-Metalium execution or hardware validation.

## StructuredBench Report Fields

A future TT-Metalium report should emit `structuredbench.v1` fields where
practical, including:

- `schema`
- `generated_at_utc`
- `suite`
- `workload`
- `backend`
- `device`
- `dtype`
- `items`
- `iterations`
- `warmup`
- `structured_shape`
- `elapsed_s`
- `latency_ms`
- `throughput`
- `throughput_unit`
- `max_abs_error`
- `max_rel_error`
- `rms_error`
- `scalar_reference_max_abs_error`
- `estimated_flops`
- `estimated_flops_per_s`
- `estimated_bytes_read`
- `estimated_bytes_written`
- `estimated_total_bytes`
- `effective_gb_per_s`
- `arithmetic_intensity_flops_per_byte`
- `checksum`

For the first `qmul` report:

```text
structured_shape = [N, 4]
throughput_unit = qmul/s
estimated_flops = 28 * N * iterations
estimated_bytes_read = 8 * sizeof(dtype) * N * iterations
estimated_bytes_written = 4 * sizeof(dtype) * N * iterations
```

The byte counts are logical traffic estimates for comparison. They are not
hardware-counter measurements unless a future backend explicitly reports
hardware counters separately.

## Placement Questions For TT-Metalium Maintainers

Before writing or submitting TT-Metalium code, RQM should ask maintainers:

1. Should this first live as a TT-Metalium programming example?
2. Should it remain in the external `tt-rqm-kernels` repo until a maintainer
   requests an upstream example?
3. Is there a preferred example layout for small structured numerical kernels?
4. Should the first layout be row-major `[N, 4]`, tile-aware, or both?
5. Is there a preferred way to emit benchmark data compatible with existing
   Tenstorrent examples?
6. Should a future TT-NN custom operation wait until the TT-Metalium example is
   validated on hardware?
7. Would a later TT-MLIR lowering discussion be useful once hardware behavior is
   measured?

## Non-Goals

This design is intentionally narrow. It does not propose:

- native quaternion hardware
- a new chip feature
- a new Tenstorrent datatype
- defense-first positioning
- a TT-NN wrapper before lower-stack placement guidance
- simulator results as hardware results
- hardware performance claims without real Tenstorrent hardware execution
- Tenstorrent endorsement of RQM Technologies or any RQM theory

The first target is only an ordinary floating-point tensor kernel with a
structured 4-lane operator contract.
