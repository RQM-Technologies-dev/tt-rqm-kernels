# Preregistered Wormhole qmul evidence program

This document defines the next measurement sequence. Empty charts and synthetic
values are intentionally absent from the public report.

## 1. Stable persistent benchmark

Collect at least three independent cold-start host sessions on Wormhole device
0 with the same candidate, pinned TT-Metalium commit, environment, seed, sizes,
warmups, iterations, and sample count. Preserve every raw timing, lifecycle
phase, device-health record, source commit, and candidate hash. Apply the
[preregistered stability thresholds](../stage-b-stability-methodology.md)
without discarding failed sessions as outliers.

## 2. Controlled core scaling

After Level 2, run N=4096, 65536, and 262144 at 1, 2, 4, 8, 16, 32, and 56
Tensix cores with an otherwise fixed candidate and layout. Publish raw samples,
parallel efficiency, and work allocation. Do not infer scaling from the current
size-dependent core counts.

## 3. Same-device ceilings and profiles

Use the pinned TT-Metal microbenchmarks to measure DRAM, NoC, dispatch, PCIe,
and compute-only SFPU ceilings on the same device and environment. Until those
artifacts exist, the qmul report's byte rate remains a logical traffic model,
not an observed hardware-fabric rate. Capture Device Program Profiler and Tracy
evidence before selecting one-change-at-a-time optimization ablations.

## 4. Matched CPU and application studies

Any CPU study must consume identical serialized Float32 inputs, implement the
same Hamilton product, validate the same output, name the implementation and
versions, disclose thread and affinity settings, and compare identical cold or
steady-state timing scopes. PoseStreamBench and sustained energy-to-solution
require separate preregistered acquisition protocols before measurements begin.

## 5. Upstream contribution

Use stable and profiled evidence to isolate the smallest useful TT-Metal change.
Create an RQM fork or patch only after that contribution is concrete; the pinned
local TT-Metal checkout remains a read-only measurement dependency meanwhile.
