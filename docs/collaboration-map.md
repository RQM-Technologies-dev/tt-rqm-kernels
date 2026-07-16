# Collaboration Map

This document maps the current, evidence-backed path from `tt-rqm-kernels` to
an upstream-reviewable Tenstorrent integration. The project is maintained by
RQM Technologies LLC and is not an official Tenstorrent repository or a
statement of Tenstorrent endorsement.

## Current State

`tt-rqm-kernels` has completed its initial real-hardware proof phase:

- `qmul` has a Claim Level 2 one-device release from three qualified N300
  device-0 sessions. The aggregate release is `stable_benchmark=true`; each
  individual session remains `false`.
- The multicore implementation uses Tensix compute/SFPU kernels, component-
  planar FP32 tiles, row-major work splitting, and a persistent device session.
- Device Program Profiler and Tracy evidence attribute the current reader,
  compute, and writer execution. The evidence is diagnostic and is not a
  measured-bandwidth or acceleration claim.
- `SU2ComposeBench` has a fused-only Claim Level 2 release from three qualified
  v3 N300 device-0 sessions. The aggregate is `stable_benchmark=true`; every
  source session remains `false`.
- The earlier three-session v2 fused/unfused campaign is retained but failed
  its frozen variability gates. It is historical, non-qualifying evidence and
  does not establish stable fused/unfused comparison performance.

The [qmul release report](benchmarks/wormhole-qmul.md),
[hardware evidence](benchmarks/wormhole-qmul-hardware-evidence.md), and
[operator contract](operator-contracts.md#qmul) are the primary review
surfaces.

## Why qmul Is Ready For Placement Review

The public operation is deliberately small:

```text
input a: FP32 [N, 4]
input b: FP32 [N, 4]
output:  FP32 [N, 4]
lanes:   [real, i, j, k]
operation: Hamilton product
```

It provides a compact test of fixed cross-lane dependencies,
noncommutative ordering, data movement, multicore work allocation, SFPU
arithmetic, register/L1 reuse, and potential fusion. It requires no new
datatype, compiler primitive, or hardware feature.

The repository now supplies what an upstream placement discussion needs:

- PyTorch and independent Float64 golden references;
- basis and noncommutative-order tests;
- whole-output FP32 validation with nonfinite rejection;
- three-session one-device stability evidence;
- profiler and controlled scaling diagnostics; and
- Apache-2.0 host and kernel source.

## Current Tenstorrent-Facing Decision

The immediate question is where the minimal upstream form belongs:

1. a TT-Metalium programming example; or
2. an experimental TT-NN device operation using the current
   `ProgramDescriptor` pattern, a Python binding, and a PyTorch golden
   function.

The existing standalone candidate directly creates programs, circular
buffers, kernels, runtime arguments, buffers, and a `MeshWorkload`. That is a
useful evidence harness, but it is not yet an idiomatic operation that ordinary
TT-NN callers can invoke.

There is also a layout decision for maintainers. `[N, 4]` is the public host
contract, while the qualified device path packs each quaternion component into
its own sequence of FP32 tiles. A TT-NN form could preserve logical `[N, 4]`
through an internal layout adapter or expose the device-native planar layout at
the experimental boundary.

The current placement request is tracked in
[tenstorrent/tt-metal#49887](https://github.com/tenstorrent/tt-metal/issues/49887).
It supersedes the earlier unanswered
[Tenstorrent discussion](https://github.com/tenstorrent/tt-metal/discussions/48871).
No upstream implementation PR will be proposed before maintainers answer the
placement and layout questions.

## Repository Roles

### `tt-rqm-kernels`

Remains the source of the operator contract, references, release evidence,
diagnostic reports, and experimental standalone candidate.

### `tt-metal`

Owns the placement decision and, if accepted, the minimal implementation in
Tenstorrent's current source structure. Work must occur in a separate fork and
current-`main` worktree; the pinned release checkout remains unchanged.

### `tt-awesome`

Provides discovery for the external community project. It does not imply that
the kernel is part of `tt-metal` or endorsed by Tenstorrent.

## Follow-On Work

After maintainers choose placement and layout, port only `qmul`, preserve its
golden tests, validate representative Wormhole shapes, and capture one
diagnostic profiler report for the upstream-shaped implementation. A new port
does not inherit the existing release's stability label.

Inside this repository, qmul Level 2 and fused-only H1 Level 2 are protected
baselines. H2A coefficient lowering is now the active implementation
milestone, with CPU reference, real single-core candidate source, focused N300
development probes, and pre-hardware conformance machinery.
H2B fusion, two-qubit hardware execution, and broader benchmark families remain
future work.

## Nonclaims

This collaboration path does not claim CPU acceleration, measured hardware
bandwidth, application speedup, energy efficiency, dual-device scaling, stable
fused/unfused SU2 comparison, H2 designated conformance, or Tenstorrent
endorsement.
