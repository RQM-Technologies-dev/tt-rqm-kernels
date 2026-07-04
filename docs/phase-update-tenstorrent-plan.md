# Phase Update Tenstorrent Backend Plan

## Purpose

This document defines a conservative future Tenstorrent backend plan for the
existing StructuredBench `phase_update` workload.

Tracker issue:
<https://github.com/RQM-Technologies-dev/tt-rqm-kernels/issues/10>

`phase_update` is not the first lower-stack target. The first target remains
`qmul` over `[N, 4]` structured tensors. This plan exists so the next signal,
imaging, wave, and sensing benchmark lane has a clear contract before any
Tenstorrent-specific code is written.

## Operator Contract

StructuredBench currently models phase integration and state generation as:

```text
phase_update(phase, angular_rate, amplitude, dt) -> state

phase:        [N] float32
angular_rate: [N] float32
amplitude:    [N] float32
dt:           scalar float
state:        [N, 2] float32
```

The output lane order is:

```text
state[..., 0] = amplitude * cos(next_phase)
state[..., 1] = amplitude * sin(next_phase)
```

where:

```text
next_phase = wrap_phase(phase + angular_rate * dt)
```

The current CPU/PyTorch reference wraps phase values into
`[-pi, pi)` through `tt_rqm_kernels.phase_ops.wrap_phase`.

## Why This Workload Matters

`phase_update` is a compact structured workload below full applications and
above scalar math. It is useful because it combines:

- streamed phase integration
- periodic wrapping
- sin/cos state generation
- amplitude scaling
- `[N] -> [N, 2]` structured output expansion

This pattern appears in engineering workloads such as wireless phase tracking,
radar/sonar-like signal processing, imaging, optical phase, audio, wave
simulation, sensing, scientific computing, and physical-AI state streams.

The claim is about benchmark structure and validation, not about a special
hardware primitive or domain-specific theory.

## Reference Validation

Any future backend candidate should compare against:

- `tt_rqm_kernels.backends.torch_backend.phase_update`
- `tt_rqm_kernels.phase_ops.integrate_phase`
- `tt_rqm_kernels.phase_ops.phase_to_unit_vector`

Validation should report:

- max absolute error
- max relative error
- RMS error
- checksum
- output shape `state=[N, 2]`
- dtype and item count

Tolerances should account for backend differences in transcendental
implementations. A candidate can pass correctness while still showing small
sin/cos differences compared with PyTorch.

## StructuredBench Report Discipline

Backend reports should keep the `structuredbench.v1` fields already used by the
CPU/PyTorch report:

- schema
- backend
- device
- dtype
- seed
- workload
- items
- iterations
- warmup
- structured shape
- latency
- throughput
- numerical error fields
- estimated FLOPs/sec
- effective GB/sec
- arithmetic intensity
- checksum

For `phase_update`, StructuredBench currently uses a simple estimate of 6
reported operations per item. The report should keep noting that this workload
is transcendental-heavy and should not be interpreted like a pure fused
multiply-add kernel.

## Future Backend Path

Recommended order:

1. Keep CPU/PyTorch as the reference.
2. Prove lower-stack `qmul` first through TT-Lang, tt-emule, TT-Metalium, or
   maintainer-guided placement.
3. Add a candidate protocol for `phase_update` only after the `qmul` candidate
   path is stable.
4. Evaluate whether `phase_update` belongs in TT-Metalium examples, TT-NN
   custom operations, TT-MLIR lowering discussion, or a separate benchmark
   package.

This document intentionally does not add TT-Metalium or TT-NN source code.

## Placement Questions

Questions for a later Tenstorrent discussion:

- Should a phase-update workload live as a TT-Metalium programming example, a
  TT-NN custom-op candidate, or only as an external benchmark?
- Is there a preferred pathway for workloads with sin/cos and wrapping?
- Should the `[N] -> [N, 2]` output be represented as lane-packed state or two
  aligned output tensors?
- Are transcendental-heavy workloads useful in Tenstorrent benchmark suites, or
  should they stay separate from early custom-kernel examples?
- Would TT-MLIR lowering be useful later for fusing phase integration,
  wrapping, and state generation?

## Non-Goals

- No TT-Metalium implementation in this milestone.
- No fake simulator, emulation, or hardware output.
- No claim that CPU/PyTorch timing predicts Tenstorrent hardware performance.
- No native phase datatype or new hardware feature request.
- No Tenstorrent endorsement claim.
- No defense-first framing.
- No speculative physics claims.
