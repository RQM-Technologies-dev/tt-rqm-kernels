# SU2ComposeBench Level 2 Stability Methodology v3

## Status and claim boundary

This file defines the v3 measurement foundation. The accompanying
preregistration is `frozen_before_designated_session_1`: it binds the clean
candidate source identity, final repeat counts, host contract, and retained
non-designated pilot packages. No designated session has been collected.

Level 2 asks one question: is the fused H1 implementation stable on one
Wormhole device? Its gates are fused whole-output correctness, fused timing
stability, provenance, lifecycle, device health, cache isolation, and frozen
host controls. Unfused timing and the fused/unfused ratio are diagnostics only.
Stable matched-scope comparison belongs to Level 3.

The eventual public Level 2 claim is limited to:

> SU2ComposeBench demonstrates stable one-device performance for fused
> time-ordered SU(2) composition across three designated Wormhole device-0
> sessions.

It does not claim acceleration, stable fused/unfused comparison, CPU
superiority, measured bandwidth, energy efficiency, H2 device-side lowering,
or Tenstorrent endorsement.

## Isolated fused execution

`benchmark_mode=fused_stability` constructs only the fused reader, compute, and
writer workload. It runs fused conformance, five fused warmups, ten measured
fused samples, fused whole-output validation, and no unfused workload. Unfused
outputs, samples, dispatch metadata, and ratios must be absent rather than
filled with synthetic values. The device arithmetic kernels are unchanged;
the new candidate narrows the host measurement surface.

Each measured sample enqueues its frozen repeat count nonblocking and performs
one `Finish` synchronization after all repeats. The raw enqueue-through-finish
duration is recorded before normalization. Every raw sample must target 25-50
ms. Per-chain samples are derived only by dividing the retained raw duration by
the frozen repeat count.

## Pilot and freeze sequence

The disclosed initial pilot repeat plan targets 37.5 ms using the retained v2
calibration report. It is not designated evidence and may be revised between
pilots. Three separate non-designated pilots were assessed for:

- exact eight-case order and deterministic inputs;
- ten complete raw samples after five warmups;
- 25-50 ms raw duration for every sample;
- within-session dispersion and cross-session median deviation;
- an absolute readiness ceiling of 10%, with 5% preferred;
- whole-output correctness and nonfinite checks;
- identical candidate/runtime and frozen host identity;
- distinct empty-start `TT_METAL_CACHE` roots.

If any case exceeds 10%, improve the harness and repeat the non-designated pilot
sequence. Do not expand a case threshold. The retained pilots passed the
preferred 5% gate, and the frozen preregistration hash-binds their packages,
candidate/source/runtime identity, and final repeat counts.

## Designated session contract

After freeze, collect exactly three sessions named:

```text
su2-v3-level2-session-1
su2-v3-level2-session-2
su2-v3-level2-session-3
```

They require separate collector invocations and host processes, complete
device close, distinct empty runtime-cache roots, fresh pre/post health and
host-state records, exact command retention, full stdout/stderr, all raw fused
samples, whole-output correctness, and a hashed session manifest. A failed or
noisy designated session is retained and cannot be replaced.

The frozen Level 2 thresholds default to 5% for both within-session dispersion
and cross-session median deviation. No case-specific threshold may exceed 10%,
and no threshold may be derived from designated data.

## Host and cache controls

The collector records inherited and requested CPU affinity, process nice value,
`/proc/loadavg`, CPU model, CPU governors and frequencies, NUMA nodes, relevant
TT-Metal environment variables, and explicit profiler/watcher/debug state. The
candidate may be bound to a fixed CPU mask. Frozen host fields must remain
identical within and across designated sessions.

Every session begins with a new empty `TT_METAL_CACHE` directory. Compilation
and loading occur during setup or warmup, outside measured samples. The cache
path and a post-session file inventory containing sizes and SHA-256 hashes are
retained. Cache roots may not be reused.

## Commands

Build the candidate, then collect one pilot with:

```bash
python scripts/collect_su2_compose_v3_pilot.py \
  --command /absolute/path/to/tt_rqm_metalium_su2_compose_candidate \
  --cpu-affinity 2-5 \
  --session-id su2-v3-pilot-1
```

After three separate invocations, assess them with:

```bash
python scripts/assess_su2_compose_v3_pilots.py \
  path/to/pilot-1/session-manifest.json \
  path/to/pilot-2/session-manifest.json \
  path/to/pilot-3/session-manifest.json
```

The assessment always retains `stable_benchmark=false` and
`qualification_passed=false`; `ready_to_freeze_v3` is only a pilot-readiness
decision.

The first N300 implementation and pilot pass is recorded in the
[v3 foundation audit](../reports/tt_hardware_su2_compose_v3_foundation_audit.md).
