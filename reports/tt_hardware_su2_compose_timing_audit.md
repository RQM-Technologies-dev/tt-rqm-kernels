# SU2ComposeBench Stability Timing Audit

This audit fixes the timing boundary used by designated H1 stability sessions.
It introduces no new hardware result.

- Each session is one cold-start host process and one device-0 lifecycle.
- All eight preregistered `(B,K)` cases run in their registered order.
- Each case performs two warmup pairs followed by ten measured pairs.
- Pair order alternates fused-first and unfused-first by sample index.
- Candidate samples cover prepared enqueue through device completion. Setup,
  allocation, H2D, D2H, program build, and lifecycle times remain separate.
- Batch samples are divided only by the preregistered repeat count. No sample is
  discarded, retried, winsorized, or replaced after inspection.
- Individual reports remain `stable_benchmark=false`. Only the deterministic
  three-session qualifier may support an aggregate Level 2 release.

Within-session dispersion is `(nearest-rank p95 - median) / median`.
Cross-session deviation is `abs(session median - median across sessions) /
median across sessions`. Fused timings, unfused timings, and paired
fused/unfused ratios must all pass the frozen per-case limits.
