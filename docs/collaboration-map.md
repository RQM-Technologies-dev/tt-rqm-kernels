# Collaboration Map

This document maps how `tt-rqm-kernels` can fit into the Tenstorrent ecosystem
as an independent RQM Technologies LLC project.

The public frame is structured computation on open accelerators: compact tensor
operators for rotation, phase, orientation, geometry, wave state, and scientific
workloads represented inside ordinary floating-point tensors.

This is not an official Tenstorrent repository unless and until accepted or
co-developed by Tenstorrent.

## Current Assets

`tt-rqm-kernels` already provides:

- CPU/PyTorch reference kernels for quaternion, rotor, inverse, normalization,
  dot-product, and phase/orientation utilities
- independent scalar reference checks for small deterministic correctness
  samples
- StructuredBench reports with latency, throughput, numerical error, estimated
  FLOPs/sec, effective GB/sec, and arithmetic intensity
- an optional TT-Lang functional simulator `qmul` prototype for `[N, 4]`
  row-major float32 tensors
- Tenstorrent-facing documentation for operator contracts, benchmark shape, and
  first integration questions
- CI that runs tests and a lightweight CPU/PyTorch smoke benchmark

The TT-Lang result is a simulator result only. It validates kernel logic and
report shape, not hardware performance.

## Why StructuredBench Is The Reusable Asset

StructuredBench is the part of the repo that can become useful beyond this
specific quaternion reference library. It defines a compact benchmark class
between scalar elementwise operations and large matrix multiplication:

```text
ordinary float tensors
-> structured 4-lane values
-> fixed cross-lane operators
-> correctness checks
-> backend-comparable reports
```

The first target is `qmul` over `[N, 4]` tensors. It is small enough to verify
carefully, but structured enough to exercise fixed multiply/add/sign patterns,
data movement, cross-lane dependencies, register reuse, fusion opportunities,
and arithmetic-intensity reporting.

The same report schema can compare:

- CPU/PyTorch reference output
- TT-Lang functional simulation
- future TT-Metalium kernels
- future TT-NN custom-operation wrappers
- future compiler-lowering experiments
- future hardware runs on Tenstorrent systems

## Collaboration Lanes

### tt-awesome

Goal:

- make `tt-rqm-kernels` discoverable as a community structured-kernel benchmark
  project

Useful placement:

- primary category: `kernels`
- secondary category: `research`
- hardware: `ttsim`, because the current Tenstorrent-adjacent proof point is
  TT-Lang simulation

### TT-Lang

Goal:

- keep the current simulator `qmul` path as the fastest bridge from PyTorch
  reference code to Tenstorrent-style custom-operation logic

Current status:

- `qmul` runs through the TT-Lang functional simulator
- output is compared against CPU/PyTorch and scalar references
- report output is compatible with `structuredbench.v1`

Next useful question:

- whether the simulator layout should remain row-major `[N, 4]` or evolve
  toward a tile-aware layout before lower-stack work

### TT-Metalium

Goal:

- implement the first real lower-stack `qmul` backend for `[N, 4]` tensors

Minimal useful target:

```text
input a: [N, 4] float tensor
input b: [N, 4] float tensor
output:  [N, 4] float tensor
operation: Hamilton product
validation: compare against CPU/PyTorch and scalar references
metrics: throughput, latency, numerical error, FLOPs/sec, GB/sec, arithmetic intensity
```

This should follow maintainer guidance from the existing `tenstorrent/tt-metal`
Discussion before a PR or example is opened.

### TT-NN

Goal:

- expose structured kernels in a way ordinary Tenstorrent developers can call
  after a lower-stack `qmul` proof exists

Potential path:

- add a Python-facing wrapper with a clear golden/reference comparison
- keep CPU/PyTorch as the correctness source
- avoid adding a TT-NN wrapper before placement guidance is clear

### TT-MLIR / TT-Forge

Goal:

- explore whether structured operators should lower as fused kernels instead of
  expanding into scalar multiply/add operations

Useful question:

```text
Should qmul lower as a fused structured operator rather than scalar expansion?
```

This discussion should come after working backend evidence, not before it.

### Tenstorrent Cloud

Goal:

- eventually publish a real Tenstorrent hardware report

Requirements:

- distinguish CPU reference, TT-Lang simulation, emulation if used, and hardware
  execution
- include exact commands and methodology
- avoid presenting sample outputs as stable hardware performance claims

## Current Ask

The immediate ask is ecosystem placement, not endorsement:

- list `tt-rqm-kernels` in `tt-awesome` as a community project
- keep the project framed as structured-kernel benchmarking
- use maintainer feedback to choose the right path for a minimal TT-Metalium
  `qmul` example

## Non-Goals

`tt-rqm-kernels` is not asking Tenstorrent to:

- add a native quaternion datatype
- add a new chip feature
- treat quaternion math as special hardware
- endorse RQM Technologies theory or research claims
- frame the project as defense-first
- accept hardware-performance claims from CPU or simulator outputs

Defense is a downstream application area. The public lead is lower-stack
structured numerical infrastructure for robotics, graphics, wireless, imaging,
wave simulation, physical AI, scientific computing, and signal processing.
