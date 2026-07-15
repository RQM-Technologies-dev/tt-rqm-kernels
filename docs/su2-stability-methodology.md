# SU2ComposeBench Stability Methodology

> Historical v1/v2 record. These campaigns are closed and must not be rerun or
> repaired. New Level 2 work follows the separate
> [v3 methodology](su2-stability-methodology-v3.md).

The historical v1 methodology was frozen before its designated session 2. The
retained-candidate v2 methodology was frozen before designated session 1. Both
apply to the same eight fused/unfused H1 cases on Wormhole device 0; the v2
contract binds the exact candidate, source tree, inputs, profiler decision,
compiler, and runtime.

## Session contract

Level 2 requires exactly three designated cold-start host sessions. Each uses
the same candidate, execution-source commit, pinned TT-Metal commit,
compiler/runtime identity, deterministic inputs, case order, device scope,
clock policy, two warmup pairs, and ten measured pairs. Pair order alternates
fused-first and unfused-first. Failed or noisy designated sessions remain in
the evidence set; none may be replaced after inspection.

Every individual report remains `stable_benchmark=false`.

## Numerical gates

For fused timing, unfused timing, and the paired `fused/unfused` ratio:

```text
within-session dispersion = (nearest-rank p95 - median) / median

cross-session deviation =
  abs(session median - median across sessions) / median across sessions
```

The per-case, per-metric limit is `max(5%, 2 * calibration dispersion)`. V1
used its first session as the anchor. V2 uses the disclosed pre-campaign
candidate experiment and never treats that experiment as a designated
session. Exact values are stored in the
[v1 preregistration](../benchmarks/manifests/su2-compose-stability-preregistration.json)
and [v2 preregistration](../benchmarks/manifests/su2-compose-stability-preregistration-v2.json).
Both absolute paths must pass because a stable ratio can hide common-mode
drift; the paired ratio is also required as a secondary comparison metric.

## Fail-closed evidence

A session is invalid after any collection, correctness, nonfinite, device
health, throttling, lifecycle, timing, sample-retention, provenance, input, or
source-cleanliness failure. The v2 collector retains the command, candidate
hash, environment, input hashes, health snapshots, stdout, stderr, reports,
collection status, and hashes of every session artifact.

The deterministic qualifier recomputes all gates. The v2 campaign retained all
three designated sessions, but five cases failed one or more frozen
within-session or cross-session limits. Its
[qualification artifact](../benchmarks/processed/wormhole-su2-compose-stability-qualification.json)
therefore records `qualification_passed=false` and
`stable_benchmark=false`. A future aggregate Level 2 manifest would have to
hash-bind a passing three-session qualification; manually setting
`stable_benchmark=true` cannot qualify a release.
