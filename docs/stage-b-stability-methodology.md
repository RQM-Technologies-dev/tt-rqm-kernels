# Stage B qmul stability methodology

Status: preregistered before the first persistent-device hardware sample.
Scope: one Wormhole device, logical device 0, Float32 qmul only.

This document fixes the rule for a future `stable_benchmark=true` decision. It
does not confer that label on the first persistent-device report and it does
not define an acceleration or CPU-comparison methodology.

## Evidence used to set thresholds

The only empirical input is the already-committed first Stage B report,
`reports/tt_hardware_qmul_stage_b_performance.json`. For each size, dispersion
is `(nearest-rank p95 - median) / median` over its ten recorded device samples:

| N | committed median s | committed p95 s | observed dispersion | preregistered limit |
|---:|---:|---:|---:|---:|
| 4096 | 0.001672700 | 0.001760370 | 5.2412% | 10.4825% |
| 65536 | 0.002140350 | 0.002165300 | 1.1657% | 5.0000% |
| 262144 | 0.004224715 | 0.004268190 | 1.0291% | 5.0000% |

The limit is `max(5%, 2 * observed dispersion)`. The 5% floor avoids claiming
false precision from ten samples; doubling the observed spread allows a
future independent session to vary without changing the rule after seeing its
result.

## Required persistent sessions

A later stability qualification must collect at least three independent,
cold-start host sessions. Each session must use the persistent protocol and:

- create Wormhole device 0 exactly once and close it exactly once;
- run N=4096, 65536, and 262144 in that order;
- use seed 0, five warmups, 30 iterations per sample, and ten samples per size;
- preserve whole-output Float64-golden validation and all provenance gates;
- keep the pinned TT-Metalium commit, candidate binary hash, clock/power mode,
  and host configuration fixed; and
- record device health before and after the session.

For every size in every session, within-session dispersion must be no greater
than the size-specific preregistered limit. For every size, each session
median must also be within that same relative limit of the median across the
three session medians. Any thermal fault, device reset, validation failure,
non-finite result, provenance mismatch, lifecycle mismatch, or timing-integrity
failure invalidates the session rather than being discarded as an outlier.

Only a new report that passes all of those gates may set
`stable_benchmark=true`. The initial persistent conformance and performance
artifacts must remain `false` even if their observed dispersion is small.

## Explicit exclusions

This policy does not compare against CPU, PyTorch, TT-Lang, tt-emule, the
scalar RISC-V baseline, or the second side of the N300. A future acceleration
claim requires a separately preregistered, timing-scope-compatible comparison
baseline.
