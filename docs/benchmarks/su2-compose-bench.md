# Fused Time-Ordered SU(2) Composition on Tenstorrent Wormhole

> **RQM runs quantum Hamiltonian simulations on Tenstorrent.**

The first implementation targets fused, time-ordered SU(2) evolution on
Wormhole using CPU-lowered FP32 evolution operators. A later stage will lower
Hamiltonian coefficients on device.

The initial fused and unfused candidates have passed N300 device-0
conformance. The result is Claim Level 0 silicon evidence and remains
`performance_eligible=false`, `stable_benchmark=false`. See the
[whole-output report](../../reports/tt_hardware_su2_compose_conformance.md) and
[architecture audit](../../reports/tt_hardware_su2_compose_architecture_audit.md).

## Problem Definition

For `B` independent trajectories and `K` piecewise-constant steps, H1 consumes
rotors `[B,K,4]` and complex phase pairs `[B,K,2]`. It returns one rotor and
phase per trajectory in exact `K-1 ... 0` multiplication order.

The inputs include varying, noncommuting Hamiltonians. Alternating x- and
y-axis rotations make an accidental order reversal fail visibly.

## Planned Hardware Comparison

The unfused baseline performs `K-1` qmul-plus-phase dispatches and moves every
intermediate accumulator through DRAM. The fused candidate performs one
dispatch and retains accumulator tiles in Tensix-local storage until the final
write. Both paths use one Wormhole device and the same serialized input.

## Correctness

Two independent CPU oracles are required: complex128 matrix exponentiation and
Float64 quaternion-plus-phase composition. Hardware output will be checked in
full against the exact serialized FP32 inputs; no primary output is
renormalized.

Published diagnostics will include matrix and state-vector error, rotor and
phase norm drift, unitarity, determinant and global-phase consistency, Bloch
norm drift, error versus chain length, and failure/nonfinite counts.

## Performance Methodology

The exact cases, sample counts, timing boundaries, traffic formulas, claim
gates, and nonclaims are fixed in the
[machine-readable preregistration](../../benchmarks/manifests/su2-compose-preregistration.json).
No chart or performance number will be added before real hardware evidence is
committed.

## Limitations And Nonclaims

H1 composes pre-lowered evolution operators. It is a real stage of a quantum
Hamiltonian simulation pipeline, but it is not yet device-side coefficient
lowering. The first hardware sample will remain `stable_benchmark=false` and
will not support an acceleration, CPU-comparison, bandwidth, energy,
dual-device, or Tenstorrent-endorsement claim.
