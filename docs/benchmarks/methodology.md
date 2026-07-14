# Preregistered Wormhole qmul evidence program

This document defines the measurement sequence and records its July 2026
execution. Empty charts and synthetic values remain absent.

## 1. Stable persistent benchmark — passed

Collect at least three independent cold-start host sessions on Wormhole device
0 with the same candidate, pinned TT-Metalium commit, environment, seed, sizes,
warmups, iterations, and sample count. Preserve every raw timing, lifecycle
phase, device-health record, source commit, and candidate hash. Apply the
[preregistered stability thresholds](../stage-b-stability-methodology.md)
without discarding failed sessions as outliers.

The three designated cold-start sessions passed every preregistered gate. The
deterministic qualification artifact is
[`wormhole-qmul-stability-qualification.json`](../../benchmarks/processed/wormhole-qmul-stability-qualification.json).
Only that artifact sets `stable_benchmark=true`; all three individual session
reports remain `false`.

## 2. Controlled core scaling — completed diagnostic

The executable matrix is N=4096 at 1/2/4 cores and N=65536 and 262144 at
1/2/4/8/16/32/56 cores. N=4096 contains only four component tiles, so larger
core counts would allocate idle cores and are prohibited. The diagnostic run
records requested and actual cores, both work groups, imbalance, raw samples,
parallel efficiency, and whole-output correctness in the
[processed scaling report](../../benchmarks/processed/wormhole-qmul-core-scaling.md).

## 3. Same-device ceilings and profiles — completed with limits

Use the pinned TT-Metal microbenchmarks to measure DRAM, NoC, dispatch, PCIe,
and compute-only SFPU ceilings on the same device and environment. Until those
artifacts exist, the qmul report's byte rate remains a logical traffic model,
not an observed hardware-fabric rate. Capture Device Program Profiler and Tracy
evidence before selecting one-change-at-a-time optimization ablations.

Device Program Profiler and Tracy evidence now cover N=65536 and N=262144.
Pinned same-device microbenchmarks produced usable PCIe, host-timed NoC,
adjacent-core DRAM-read, dispatch, and BFP8 FPU measurements. The pinned suite
does not provide an FP32 SFPU ceiling, and its circular-buffer stall, NoC-wait,
and SFPU-utilization counters are not observable. These limits and all failed
attempts are preserved in the
[profiler and ceiling report](../../benchmarks/processed/wormhole-qmul-profiler-and-ceilings.md).

## 4. Device parity, initialization, and saturation — completed diagnostics

Device 1 passed N=128 conformance and the three-size persistent parity run with
the same diagnostic binary. First-use order tests attribute the elevated first
H2D and warmup cost to first submission/dispatch initialization rather than
input size or program construction. The size sweep passed from N=1024 through
N=1048576 after a 48 MiB largest-case buffer preflight. These are diagnostic
artifacts, not stability, dual-device, acceleration, or application claims.

## 5. Matched CPU and application studies — not measured

Any CPU study must consume identical serialized Float32 inputs, implement the
same Hamilton product, validate the same output, name the implementation and
versions, disclose thread and affinity settings, and compare identical cold or
steady-state timing scopes. PoseStreamBench and sustained energy-to-solution
require separate preregistered acquisition protocols before measurements begin.

## 6. Upstream contribution — not started

Use stable and profiled evidence to isolate the smallest useful TT-Metal change.
Create an RQM fork or patch only after that contribution is concrete; the pinned
local TT-Metal checkout remains a read-only measurement dependency meanwhile.
