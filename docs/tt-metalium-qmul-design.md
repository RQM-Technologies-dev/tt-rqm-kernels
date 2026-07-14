# TT-Metalium qmul Design

This document describes the immutable scalar RISC-V Stage A candidate and the
separate multicore/SFPU Stage B candidate for `tt-rqm-kernels`.

The scalar candidate has passed N300 silicon conformance. The active development
target moves Float32 Hamilton arithmetic into the Tensix compute path on one
Wormhole device.

## Purpose

The first TT-Metalium target is quaternion multiply over structured `[N, 4]`
floating-point tensors.

This workload is useful because it is compact but not scalar-trivial:

- every output lane depends on every input lane
- the multiply/add/sign pattern is fixed and easy to audit
- correctness can be checked against existing CPU/PyTorch and scalar references
- benchmark output can reuse the existing StructuredBench report shape

The scalar version is explicitly `scalar_riscv_correctness_baseline` with
`performance_eligible=false`. It is suitable for Stage A silicon conformance,
not acceleration claims. The separate Stage B implementation identifies itself
as `multicore_tensix_sfpu_qmul`; it became `performance_eligible=true` only
after hardware conformance and architecture/hash audits passed.

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

The diagnostic comparison includes the existing CPU/PyTorch implementation:

```text
tt_rqm_kernels.quaternion_ops.qmul
```

The acceptance source is an independent Hamilton-product calculation over all
exact float32 inputs promoted to float64. The first-eight scalar diagnostic uses:

```text
tt_rqm_kernels.backends.scalar_reference.qmul_scalar
```

Recommended validation path:

1. Generate deterministic `[N, 4]` input tensors.
2. Build the independent float64 golden result from the serialized float32 inputs.
3. Run the TT-Metalium `qmul` candidate.
4. Compare elementwise max absolute error, max relative error, and RMS error.
5. Reject every non-finite value or whole-output tolerance failure; retain the
   first-eight scalar comparison as a diagnostic.
6. Include identity and basis multiplication cases in unit-level validation.
7. Record dtype, input size, seed, and backend identity in the report.

The TT-Lang simulator report is useful as a pre-hardware logic check, but it is
not a substitute for TT-Metalium execution or hardware validation.

For an external TT-Metalium executable, the intended bridge is the StructuredBench
`external-qmul` backend. It writes deterministic `a.bin`, `b.bin`, and
`manifest.json` files, runs a candidate command, then validates `out.bin` and
`metrics.json` against the same CPU/PyTorch and scalar references. The included
`scripts/qmul_external_reference.py` command is only a CPU/PyTorch protocol
reference, not a hardware backend.

The implementation area is `experimental/tt_metalium_qmul/`. It contains the
scalar TT-Metalium host/kernel source validated under tt-emule and N300, plus
the distinct Stage B host, reader, compute/SFPU, and writer sources.

The current package includes:

- `check_environment.py` for detecting a local `tt-metal` checkout
- `build_candidate.py --candidate {scalar,multicore}` for building the selected
  target against TT-Metalium; it defaults to `scalar`
- scalar RISC-V candidate source and a Docker/tt-emule execution wrapper
- a planar Float32 multicore target using device 0, row-major work splitting,
  DMA-only reader/writer kernels, and a compute/SFPU Hamilton kernel
- `validate_candidate.py` as the wrapper around `scripts/validate_qmul_candidate.py`

The scalar implementation is Stage A correctness evidence only.

## Stage B Layout and Work Split

The Stage B host converts `[N,4]` AoS inputs into four padded component planes
per operand. Each plane is stored as 32x32 Float32 tiles. `N=128` therefore uses
one component tile with 896 zero-padded rows; readback interleaves only the
first 128 rows into `out.bin`.

The candidate creates a unit mesh for device 0 and calls
`split_work_to_cores(grid, component_tiles, true)`. Component tiles are assigned
row-major to `min(component_tiles, available Tensix cores)`. Each selected core
runs three roles:

- reader RISC-V: DMA four A and four B component tiles into CBs 0-7;
- compute/SFPU: evaluate each output component as one product plus three signed
  product accumulations through three FP32 destination registers, then pack CBs
  16-19;
- writer RISC-V: DMA the four output component tiles to planar DRAM.

Only the compute kernel and its SFPU helper contain Hamilton arithmetic. The
data-movement kernels contain addressing and DMA control only.

## Stage B Hardware Evidence

The protected N=128 candidate report validates 512 values with zero failing or
non-finite values and maximum absolute error `9.686581936563243e-08`. The
architecture audit confirms device 0 only, an 8x7 compute grid, DMA-only
reader/writer roles, compute/SFPU arithmetic, and projected Stage B core counts
of 4, 56, and 56.

The first official sweep used N=4096/65536/262144, 30 measured iterations, 5
warmups, 10 repetitions, and seed 0. Each result retains all ten setup, device,
and end-to-end samples plus median and nearest-rank p95. Whole-output validation
passed at every size. The report remains `stable_benchmark=false`, so these
measurements establish a reproducible first sample, not an acceleration claim.

- [candidate conformance](../reports/tt_hardware_qmul_stage_b_candidate_conformance.md)
- [architecture and eligibility audit](../reports/tt_hardware_qmul_stage_b_architecture_audit.md)
- [first performance sample](../reports/tt_hardware_qmul_stage_b_performance.md)

## StructuredBench Report Fields

A TT-Metalium report emits additive `structuredbench.v1` fields, including:

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
- `correctness`
- `timing` with setup/device/end-to-end median and p95
- `provenance`
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
