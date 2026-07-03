# StructuredBench Specification

This document is written for Tenstorrent engineers evaluating whether `tt-rqm-kernels` is a useful external structured-kernel benchmark suite.

## What StructuredBench Is

StructuredBench is a reproducible benchmark suite for structured tensor operators where each logical value is stored inside ordinary floating-point tensor lanes.

The first convention is quaternion layout:

```text
[N, 4] = [real, i, j, k]
```

The current backend is CPU/PyTorch. The report schema is designed so future TT-Metalium and TT-NN implementations can be compared against the same correctness and benchmark fields.

## Current Status

- current backend: CPU/PyTorch
- current benchmark reports are sample reference outputs
- Tenstorrent backend is not implemented yet
- first requested maintainer guidance is placement for a minimal TT-Metalium `qmul` example

## What This Lets Tenstorrent Demonstrate

StructuredBench gives Tenstorrent a small public workload for structured numerical kernels, not another LLM benchmark. It shows how ordinary floating-point tensors can carry rotation, phase, orientation, direction, and geometric state without a new datatype or hardware feature.

The first path is intentionally narrow: CPU/PyTorch reference results, scalar correctness checks, then a future TT-Metalium `qmul` comparison for `[N, 4]` tensors. That lets Tenstorrent demonstrate a custom kernel path for 4-lane structured values with fixed cross-lane dependencies, then compare throughput, latency, numerical error, FLOPs/sec, effective GB/sec, and arithmetic intensity against the reference report.

## Why Structured Tensor Kernels Matter

Many accelerator benchmarks focus on scalar elementwise math or large matrix multiplication. Between those levels is a useful class of structured numerical operators:

- geometric rotation and orientation updates
- phase tracking
- wave-state updates
- signal and imaging transforms
- simulation and scientific-computing kernels
- physical AI state representations

These values can live inside dense tensor layouts while still requiring structured algebra. That makes them relevant to accelerator hardware without requiring a new hardware primitive.

## Why `qmul` Is Not Just Arbitrary Elementwise Math

Quaternion multiplication is a structured 4-lane operation. Every output lane depends on all lanes from both inputs with a fixed sign pattern:

```text
out.w = a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z
out.x = a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y
out.y = a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x
out.z = a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w
```

This is small enough to understand and validate, but structured enough to exercise kernel fusion, data layout, register reuse, and vector-lane handling.

## Why The Workload Stresses Useful Accelerator Behaviors

StructuredBench workloads are compact but not trivial:

- `qmul` stresses fixed 4-lane cross-lane dependency patterns.
- `qrotate_vector` stresses streamed structured rotation using two Hamilton products.
- `qnormalize` stresses 4-lane reduction plus reciprocal/division and output scaling.
- `qinverse` stresses norm-squared, conjugate, reciprocal/division, and identity residual stability.
- `phase_update` stresses phase integration and transcendental-heavy `[cos, sin]` state generation.

These patterns are common below full applications and above scalar math.

## Why `[N, 4]` Is The First Layout

`[N, 4]` is the smallest practical layout for the initial benchmark:

- it directly represents one quaternion per row
- it maps cleanly onto ordinary floating-point tensors
- it avoids a custom dtype or object representation
- it is easy to validate against independent scalar reference code
- it can be flattened from richer batch, spatial, sequence, or stream shapes

Later backend work can explore tile-specific layouts, but `[N, 4]` is the right first public contract.

## Reported Metrics

StructuredBench reports:

- throughput
- latency
- max absolute error
- max relative error
- RMS error
- stability metrics where relevant
- scalar reference spot-check error
- estimated FLOPs
- estimated FLOPs/sec
- estimated bytes read
- estimated bytes written
- estimated total bytes
- effective GB/sec
- arithmetic intensity in FLOPs/byte

The hardware metrics are estimates. They are intended for comparison and placement discussion, not as hardware-counter measurements.

## What A TT-Metalium Result Should Compare Against

A first TT-Metalium result should compare against:

- CPU/PyTorch StructuredBench JSON report
- scalar reference spot checks on small deterministic samples
- operator contracts in `docs/operator-contracts.md`
- qmul benchmark rows for `[N, 4]` tensors

The first useful comparison should include throughput, latency, numerical error, scaling across input sizes, effective GB/sec, and arithmetic intensity.

## External qmul Candidate Harness

StructuredBench includes an `external-qmul` backend for candidate executables
that are not built into the Python package:

```bash
python -m tt_rqm_kernels.structuredbench \
  --suite qmul \
  --backend external-qmul \
  --external-command "python scripts/qmul_external_reference.py"
```

This backend is intentionally narrow. It supports only float32 `qmul` over
`[N, 4] x [N, 4] -> [N, 4]`. StructuredBench generates deterministic `a.bin`,
`b.bin`, and `manifest.json` files in a temporary work directory, exposes that
directory through `TT_RQM_EXTERNAL_QMUL_DIR`, and expects the candidate command
to write `out.bin` and `metrics.json`.

`metrics.json` must include a positive finite `elapsed_s` value measured over the
requested iteration loop. It may include `device` to label the candidate system
in the StructuredBench report.

The output is validated against the CPU/PyTorch `qmul` reference and scalar spot
checks before it is reported through the normal `structuredbench.v1` fields.
This provides a bridge for future TT-Metalium, TT-NN, or cloud-hosted candidate
runs without adding fake Tenstorrent code to this repository.

## Successful First Tenstorrent Backend Result

A successful first Tenstorrent backend result would:

1. Implement `qmul` for `[N, 4]` floating-point tensors.
2. Match CPU/PyTorch output within a documented dtype tolerance.
3. Emit `structuredbench.v1`-compatible metrics.
4. Show scaling across at least the existing `qmul` suite sizes.
5. Be placed where Tenstorrent maintainers think external examples or custom operations should live.

The proposed second target is `qrotate_vector`, because it builds on `qmul` and exercises a more complete geometric stream operation.

## Why Tenstorrent Should Care

Tenstorrent hardware should not only be evaluated on LLM inference and matmul-heavy neural networks. StructuredBench provides a compact, reproducible workload family for structured physical, geometric, signal, and simulation operators represented inside ordinary floating-point tensors.

This gives Tenstorrent engineers a small public benchmark for a workload class that can matter across robotics, graphics, wireless, imaging, wave simulation, physical AI, scientific computing, signal processing, and downstream signals processing without making defense the primary framing.
