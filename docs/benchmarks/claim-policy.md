# Benchmark claim policy

Benchmark claims advance only when the corresponding public evidence gate is
complete. A higher implementation capability does not automatically grant a
higher evidence level.

| Level | Public meaning | Minimum evidence |
|---:|---|---|
| 0 | Silicon conformance | Real-device whole-output validation and provenance |
| 1 | Qualified first performance sample | Audited architecture, raw samples, timing scopes, and nonclaims |
| 2 | Stable one-device performance | At least three independent cold-start sessions passing preregistered thresholds |
| 3 | Matched-scope baseline comparison | Named and versioned baseline with identical operation, inputs, validation, and timing boundaries |
| 4 | Application workload result | Preregistered workload contract and end-to-end application evidence |
| 5 | Reviewed upstream contribution | A concrete contribution reviewed or accepted through the upstream project |

The current Wormhole qmul release is **Level 2**. Three independent device-0
cold-start sessions passed the preregistered thresholds with identical
candidate and environment provenance. Individual reports remain
`stable_benchmark=false`; the cross-session qualification and release manifest
set it to `true`.

The release validator rejects Level 2 with fewer than three independent public
sessions or without a hashed stability-qualification artifact.
