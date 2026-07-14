# SU2ComposeBench Stability Methodology

This methodology is frozen before designated stability session 2. It applies
to the eight published fused/unfused H1 cases on Wormhole device 0.

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

The per-case, per-metric limit is `max(5%, 2 * session-1 observed
dispersion)`. Exact values are stored in the
[machine-readable preregistration](../benchmarks/manifests/su2-compose-stability-preregistration.json).
Both absolute paths must pass because a stable ratio can hide common-mode
drift; the paired ratio is also required as a secondary comparison metric.

## Fail-closed evidence

A session is invalid after any collection, correctness, nonfinite, device
health, throttling, lifecycle, timing, sample-retention, provenance, input, or
source-cleanliness failure. The v2 collector retains the command, candidate
hash, environment, input hashes, health snapshots, stdout, stderr, reports,
collection status, and hashes of every session artifact.

The deterministic qualifier recomputes all gates. A future aggregate Level 2
manifest must hash-bind the three reports and session manifests, both
preregistrations, the qualification artifact, architecture and timing audits,
and generated aggregate outputs. Manually setting `stable_benchmark=true`
cannot qualify a release.
