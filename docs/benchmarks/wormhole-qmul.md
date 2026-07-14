# Structured FP32 Quaternion Kernels on Tenstorrent Wormhole

Correctness, multicore scaling, memory efficiency, persistent latency, and
energy-to-solution are the measurement program. This first public report covers
real-device correctness, the multicore compute architecture, one persistent
timing session, and its complete limitations.

> **Qualification: Claim Level 1 — qualified first performance sample.**
>
> `performance_eligible=true`, `stable_benchmark=false`, one public session.

## Kernel contract

```text
qmul: Float32 [N, 4] x [N, 4] -> [N, 4]
lanes: [real, i, j, k]
operation: Hamilton product
model: 28 floating-point operations and 48 logical bytes per qmul
```

Host AoS values are converted into planar 32x32 Float32 tiles. A reader
data-movement kernel transfers eight component planes into circular buffers; a
Tensix compute kernel performs the Hamilton-product multiply/add/subtract
arithmetic; and a writer data-movement kernel transfers four output planes to
DRAM. Tiles are split row-major over `min(component_tiles, 56)` Tensix cores on
Wormhole device 0. The persistent candidate creates that device once for the
three-size session and closes it once.

The [architecture audit](../../reports/tt_hardware_qmul_stage_b_architecture_audit.md)
checks that arithmetic is in the compute path rather than the data-movement
kernels. The [timing audit](../../reports/tt_hardware_qmul_stage_b_persistent_timing_audit.md)
defines the synchronization and timer boundaries.

## Qualified evidence snapshot

The table is supporting evidence from one public persistent session. Throughput
uses the median prepared-workload device time. “Logical GB/s” divides the fixed
48-byte-per-qmul traffic model by that time; it is not an observation of the
DRAM, NoC, or PCIe fabrics.

| N | cores | median device ms | p95 device ms | qmul/s | logical GB/s | validated values | max abs error |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4,096 | 4 | 1.651627 | 1.724061 | 74,399,342 | 3.571 | 16,384 | 7.663e-07 |
| 65,536 | 56 | 2.101087 | 2.143976 | 935,744,212 | 44.916 | 262,144 | 1.542e-06 |
| 262,144 | 56 | 4.231194 | 4.303374 | 1,858,652,664 | 89.215 | 1,048,576 | 1.487e-06 |

Every result passed the independent Float64 Hamilton-product golden across the
whole output with `atol=1e-4`, `rtol=1e-4`, zero failing values, and zero
non-finite values.

![Throughput from the first persistent session](../../benchmarks/plots/wormhole-qmul-throughput.svg)

![Recorded setup, device, readback, and end-to-end timing scopes](../../benchmarks/plots/wormhole-qmul-timing-breakdown.svg)

![All device samples from the first persistent session](../../benchmarks/plots/wormhole-qmul-raw-samples.svg)

![Whole-output correctness evidence](../../benchmarks/plots/wormhole-qmul-correctness.svg)

## Timing and provenance

The primary time is the median of ten prepared-workload device samples. Separate
fields preserve device creation, buffer allocation, program build, H2D,
prewarm synchronization, warmup, every measured sample, D2H, cleanup, device
close, candidate-session time, and host-process end-to-end time.

| Evidence field | Value |
|---|---|
| Hardware scope | one Wormhole device, logical device 0 |
| Candidate SHA-256 | `179a5cc3e6b146a1e8c61e61ab9ab173bbc543f88181b91c8621a7e959c98ce5` |
| Execution-source commit | `3ae68815e8ac025e49f09d3797dbbac2f77245b3` |
| TT-Metalium commit | `dd2849b5bc6b7a5d38a9eafbeba31ef8d530f8d4` |
| Candidate session | 2.407865554 s |
| Device creation / close | 2.087592366 s / 0.130627294 s |
| Host process end-to-end | 3.060525127 s |

The canonical [JSON report](../../reports/tt_hardware_qmul_stage_b_persistent_performance.json),
[environment record](../../reports/tt_hardware_qmul_stage_b_persistent_environment.txt),
and [release manifest](../../benchmarks/manifests/wormhole-qmul.json) are the
sources of truth. The manifest hashes every Stage A, Stage B, and persistent
artifact used by this release.

## Reproduce and validate

Install the development dependencies and validate hashes, provenance, claim
gates, normalized data, and byte-for-byte plot regeneration:

```bash
python -m pip install -e ".[dev]"
python scripts/reproduce_wormhole_qmul.py --check
```

Regenerate the committed outputs explicitly:

```bash
python scripts/generate_benchmark_plots.py
python scripts/validate_benchmark_release.py
```

New hardware evidence is opt-in. It must name a real persistent candidate and
writes to a new timestamped directory under `benchmarks/raw`:

```bash
python scripts/reproduce_wormhole_qmul.py \
  --collect-stage performance \
  --command /absolute/path/to/tt_rqm_metalium_qmul_multicore_persistent_candidate
```

## Limits and next measurements

This report does not establish run-to-run stability, controlled core scaling,
hardware-fabric ceilings, profiler attribution, a CPU comparison, sustained
energy, PoseStreamBench performance, device-1 or dual-device behavior,
acceleration, or endorsement by Tenstorrent.

Those measurements remain explicitly “not measured yet.” Their order and
acceptance rules are defined in the [evidence program](methodology.md) and
[claim policy](claim-policy.md). In particular, Claim Level 2 requires at least
three independent cold-start sessions; the current session remains immutable
and non-stable even if later sessions qualify.
