# StructuredBench-HPC Expansion Roadmap

## Purpose

This roadmap defines how `tt-rqm-kernels` can broaden StructuredBench beyond
the current quaternion, rotor, inverse, normalization, and phase workloads
without losing its identity.

Tracker issue:
<https://github.com/RQM-Technologies-dev/tt-rqm-kernels/issues/13>

The first wedge remains `qmul` over `[N, 4]` structured floating-point tensors.
HPC expansion should happen one workload at a time, with CPU/PyTorch references
and backend-comparable reports before any Tenstorrent implementation claims.

## Admission Criteria

A new StructuredBench-HPC workload should have:

- a compact tensor contract that fits ordinary floating-point tensors
- a CPU/PyTorch reference implementation
- deterministic input generation
- independent scalar, analytical, or stability checks where practical
- clear shape and dtype rules
- documented estimated FLOPs and logical memory traffic
- `structuredbench.v1`-compatible report fields or a documented extension path
- no reliance on speculative physics or domain-specific theory
- no fake Tenstorrent backend

If a workload cannot meet these criteria, keep it as an example or design note
until the validation path is clearer.

## Current Foundation

StructuredBench already covers:

- `qmul`: fixed four-lane cross-lane Hamilton product
- `qrotate`: streamed rotor/vector rotation
- `qnormalize`: four-lane reduction and scaling
- `qinverse`: norm-squared, conjugate, reciprocal/division, and residual check
- `phase_update`: phase integration, wrapping, and `[cos, sin]` state generation

Related docs and demos now cover:

- external `qmul` candidate harnesses
- TT-Lang simulator `qmul`
- tt-emule validation planning
- ComplexTensor-to-QuaternionTensor bridge framing
- physical-AI pose stream reporting
- phase-update backend planning
- external LWT/ILWT contribution selection

## Staged Workload Families

### 1. Pose And Orientation Streams

Build from the current physical-AI pose stream demo.

Candidate StructuredBench workload:

```text
orientation = qnormalize(qmul(delta_rotor, base_orientation))
world_vector = qrotate_vector(orientation, body_vector)
```

Why it matters:

- extends `qrotate_vector` from isolated rotations to stream-style state updates
- gives robotics, sensing, graphics, and physical-AI users a concrete example
- reports unit-rotor stability and vector-norm preservation

Gate before adding to the main CLI:

- decide whether this belongs as a StructuredBench suite or example-only report
- define estimated FLOPs and logical traffic
- keep CPU/PyTorch as the reference

### 2. Phase And Magnitude Streams

Build from the existing `phase_update` workload.

Candidate extension:

```text
phase_next = wrap_phase(phase + omega * dt)
state_next = magnitude * [cos(phase_next), sin(phase_next)]
```

Why it matters:

- signal, wireless, imaging, audio, and wave-state relevance
- compact `[N] -> [N, 2]` structured output
- exposes transcendental-heavy behavior separately from pure multiply-add paths

Gate before backend work:

- keep `phase_update` after `qmul` and `qrotate_vector`
- decide tolerance policy for backend sin/cos differences
- define candidate protocol only after the `qmul` external path is proven

### 3. Small Wave Or Stencil Updates

Future candidate workload:

```text
state_next[i] = a * state[i - 1] + b * state[i] + c * state[i + 1]
```

Why it matters:

- compact scientific/HPC kernel pattern
- neighbor access and halo behavior
- useful comparison point for wave, imaging, and simulation pipelines

Gate before adding:

- write a separate design doc with boundary behavior and reference checks
- avoid claiming direct alignment with any upstream Tenstorrent spectral-element
  work unless there is a concrete integration point
- keep it independent from the quaternion/rotor claims

### 4. Compact Vector-State Updates

Future candidate workload:

```text
position_next = position + velocity * dt
velocity_next = velocity + acceleration * dt
```

Why it matters:

- simple physical state stream
- low arithmetic intensity but clear memory movement
- useful for physical-AI and simulation examples

Gate before adding:

- define a stability or conservation check that is not domain-overclaimed
- keep it compact enough to remain a benchmark kernel, not an application

### 5. Complex/Quaternion Bridge Kernels

Build from `docs/complex-quaternion-bridge.md`.

Candidate utility path:

```text
split_quaternion_lanes(q) -> (real, i, j, k)
pack_quaternion_lanes(real, i, j, k) -> q
```

Why it matters:

- helps TT-NN maintainers compare existing ComplexTensor patterns to
  QuaternionTensor-style structured values
- preserves the `[N, 4]` public benchmark contract

Gate before adding:

- start as CPU/PyTorch utility validation or docs examples
- do not create a TT-NN wrapper until lower-stack evidence or maintainer
  guidance justifies it

## Reporting Requirements

Every added workload should preserve these report properties:

- clear backend and device labels
- explicit CPU, simulator, emulation, or hardware status
- dtype, seed, item count, iterations, and warmup
- latency and throughput
- max absolute error, max relative error, and RMS error where applicable
- stability or residual metric where meaningful
- estimated FLOPs/sec, effective GB/sec, and arithmetic intensity
- checksum
- notes for transcendental-heavy or simulator-only cases

Committed reports must remain sample CPU/PyTorch reference outputs unless they
come from a real, clearly labeled simulator, emulation, or hardware run.

## Recommended Order

1. Keep `qmul` as the first lower-stack Tenstorrent target.
2. Convert the pose-stream demo into a StructuredBench suite only if its report
   contract adds value beyond the existing `qrotate` suite.
3. Mature `phase_update` only after `qmul` candidate validation is stable.
4. Add one wave/stencil design doc before implementing any wave/stencil suite.
5. Keep ComplexTensor bridge work as documentation or CPU utility validation
   until lower-stack evidence exists.

## Non-Goals

- No native quaternion hardware request.
- No new silicon feature request.
- No fake TT-Metalium, TT-NN, simulator, emulation, or hardware output.
- No claim that CPU/PyTorch timings predict Tenstorrent hardware performance.
- No speculative physics claims.
- No defense-first framing.
