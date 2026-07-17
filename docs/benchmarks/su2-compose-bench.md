# Fused Time-Ordered SU(2) Composition on Tenstorrent Wormhole

> **RQM runs fused time-ordered SU(2) evolution for two-level Hamiltonian simulation on Tenstorrent Wormhole.**

H1 lowers piecewise-constant two-level Hamiltonian coefficients into FP32
rotors and phase pairs on the CPU. Wormhole performs their ordered composition.
H2A device-side Hamiltonian coefficient lowering has a separate Claim Level 0
release. H2B now has a CPU/reference foundation and a two-program TT-Metal
candidate source that feeds H2A output directly into H1 without an intermediate
host round trip. Two non-designated H2B sessions are retained and failed before
numerical output; H1 remains a separately
protected stage and H2B inherits none of its evidence.

The fused and unfused paths passed N300 device-0 conformance before and after
the audited eligibility promotion. The historical first comparison remains
**Claim Level 1** and `stable_benchmark=false`. The current release is **Claim
Level 2: stable one-device fused performance**, established only by the three
designated v3 sessions described below; it is not an acceleration or stable
fused/unfused-comparison claim.

The historical report Markdown is hash-bound release evidence and remains
byte-for-byte unchanged. This page and the claim policy provide the current,
more precise public framing; future generated reports use the same wording.

## Stability campaign outcome

The retained v2 campaign remains historical and non-qualifying: its three
sessions failed five frozen variability gates and were not replaced. The later
fused-only v3 campaign froze its candidate, host controls, repeat counts, and
pilot evidence before collection. Its three designated N300 device-0 cold-start
sessions passed whole-output correctness, provenance, lifecycle, host/cache,
raw-duration, and 5% fused within/cross-session gates.

The historical [v2 qualification result](../../benchmarks/processed/wormhole-su2-compose-stability-qualification.json)
and [v2 preregistration](../../benchmarks/manifests/su2-compose-stability-preregistration-v2.json)
remain retained. The [v3 qualification](../../benchmarks/processed/wormhole-su2-compose-v3-stability-qualification.json)
and [Level 2 release manifest](../../benchmarks/manifests/wormhole-su2-compose-level2.json)
set `stable_benchmark=true` only for the aggregate; every source session remains
`stable_benchmark=false`.

This H1 candidate and its evidence are protected baselines. H2 work must not
alter the historical packages, replace the frozen candidate, or inherit its
stable label on a different TT-Metal baseline.

## Kernel Architecture

H1 lowers time-dependent two-level Hamiltonians into FP32 rotors and phase
pairs on the CPU. One binary then runs two matched paths on Wormhole device 0:

- **Unfused:** K-1 persistent qmul-plus-phase dispatches with DRAM ping-pong
  accumulators and runtime-argument updates.
- **Fused:** one reader-compute-writer workload that retains four rotor and two
  phase accumulator tiles in Tensix L1 across the complete chain.

Both paths use the same step-major, component-planar 32x32-tile input. All
Hamilton-product and phase arithmetic is in the compute/SFPU kernels; the
reader and writer perform only DMA and synchronization. Trajectory tiles are
split row-major across up to 56 Tensix cores.

## Problem Definition

For `B` independent trajectories and `K` piecewise-constant steps, H1 consumes
rotors `[B,K,4]` and complex phase pairs `[B,K,2]`. It returns one rotor and
phase per trajectory in exact `K-1 ... 0` multiplication order.

The inputs include varying, noncommuting Hamiltonians. Alternating x- and
y-axis rotations make an accidental order reversal fail visibly.

## First Hardware Comparison

The balanced-work cases below are exact medians from one public session. Each
case has two warmup pairs and ten measured pairs with alternating path order.
They are supporting evidence, not the headline claim.

| B | K | Tensix cores | fused median | unfused median | fused/unfused |
|---:|---:|---:|---:|---:|---:|
| 32,768 | 8 | 32 | 0.141 ms | 0.667 ms | 0.211 |
| 8,192 | 32 | 8 | 0.421 ms | 1.977 ms | 0.213 |
| 2,048 | 128 | 2 | 1.575 ms | 7.256 ms | 0.217 |
| 512 | 512 | 1 | 6.180 ms | 26.700 ms | 0.231 |

The same session also covers B=1,024/4,096/16,384/65,536 at K=128. See the
[canonical report](../../reports/tt_hardware_su2_compose_first_comparison.md),
[release manifest](../../benchmarks/manifests/wormhole-su2-compose.json), and
[processed evidence](../../benchmarks/processed/wormhole-su2-compose-summary.json).

![Fused and unfused latency](../../benchmarks/plots/wormhole-su2-compose-latency.svg)

![Fused throughput](../../benchmarks/plots/wormhole-su2-compose-throughput.svg)

## Correctness

Two independent CPU oracles are used: complex128 matrix exponentiation and
Float64 quaternion-plus-phase composition. Every hardware output was checked
against the exact serialized FP32 inputs; no primary output was renormalized.

All eight cases recorded zero failing and zero nonfinite values. The largest
fused max absolute error was `1.868e-6`, below the preregistered `1e-4`
tolerance. The report also records matrix and state-vector error, rotor and
phase norm drift, unitarity, determinant and global-phase consistency, Bloch
norm drift, and error versus chain length.

![Error and drift](../../benchmarks/plots/wormhole-su2-compose-error-drift.svg)

## Performance Methodology

The exact cases, repeat counts, timing boundaries, logical-traffic formulas,
claim gates, and nonclaims were fixed before collection in the
[machine-readable preregistration](../../benchmarks/manifests/su2-compose-preregistration.json).
The [raw paired samples](../../benchmarks/raw/su2-compose/2026-07-14-n300-device0-session-1/raw-samples.json)
and generated SVGs are deterministic products of the committed hardware
report. Validate everything without hardware using:

```bash
python scripts/reproduce_wormhole_su2_compose.py --check
```

![Raw paired samples](../../benchmarks/plots/wormhole-su2-compose-raw-paired-samples.svg)

![Timing breakdown](../../benchmarks/plots/wormhole-su2-compose-timing-breakdown.svg)

## Limitations And Nonclaims

H1 composes pre-lowered evolution operators. It is a real stage of a quantum
Hamiltonian simulation pipeline, but it is not yet device-side coefficient
lowering. The aggregate Level 2 result supports stable one-device fused
performance only. It does not support an acceleration, stable fused/unfused
comparison, CPU-comparison, measured-bandwidth, energy, dual-device, or
Tenstorrent-endorsement claim. Its frozen 5% within-session and cross-session
gates are documented in the [v3 methodology](../su2-stability-methodology-v3.md).
