# N300 SU2ComposeBench Candidate Experiment (`54b91b…`)

This page records a separate real-hardware experiment for the rebuilt
SU2ComposeBench candidate with SHA-256
`54b91bd921a67bcbda0faaafc2019bbfb931a7f1ef5cef26913d252d0f01da16`.
It is preserved for reproducibility and engineering comparison; it is not a
release, a designated stability session, or an acceleration claim.

## Scope and provenance

- Hardware: N300, Wormhole device 0 only, with health snapshots before and
  after collection.
- Candidate source: `3238299…`; TT-Metal: `dd2849…`.
- Conformance: two cases passed with one device create/close lifecycle and no
  recorded correctness or nonfinite failures.
- Performance: eight frozen H1 cases, two warmup pairs and ten measured paired
  fused/unfused samples per case, alternating path order.
- Claim state: `performance_eligible=true`, `stable_benchmark=false`.

The evidence was collected as a new candidate experiment because its candidate
hash differs from the candidate frozen in the existing Level 2 stability
preregistration. It therefore cannot serve as session 2 of that campaign.

## One-session performance observation

| B | K | Tensix cores | fused median | unfused median | fused/unfused |
|---:|---:|---:|---:|---:|---:|
| 32,768 | 8 | 32 | 0.141 ms | 0.662 ms | 0.212 |
| 8,192 | 32 | 8 | 0.418 ms | 1.962 ms | 0.213 |
| 2,048 | 128 | 2 | 1.576 ms | 7.265 ms | 0.217 |
| 512 | 512 | 1 | 6.181 ms | 26.765 ms | 0.231 |
| 1,024 | 128 | 1 | 1.574 ms | 7.233 ms | 0.218 |
| 4,096 | 128 | 4 | 1.581 ms | 7.608 ms | 0.208 |
| 16,384 | 128 | 16 | 1.586 ms | 7.858 ms | 0.202 |
| 65,536 | 128 | 56 | 3.149 ms | 15.080 ms | 0.209 |

The largest recorded fused absolute error was `1.868e-6`, within the
experiment's `1e-4` tolerance. These numbers characterize one retained
hardware run only. They do not establish stable performance, CPU superiority,
measured bandwidth, energy efficiency, application speedup, dual-device
scaling, or device-side Hamiltonian coefficient lowering.

## Retained evidence

- [Conformance package](../../benchmarks/raw/su2-compose/2026-07-15-n300-device0-candidate-54b91b-conformance-1/)
- [Conformance report](../../benchmarks/raw/su2-compose/2026-07-15-n300-device0-candidate-54b91b-conformance-1/conformance.md)
- [Performance package](../../benchmarks/raw/su2-compose/2026-07-15-n300-device0-candidate-54b91b-experiment-1/)
- [Performance report](../../benchmarks/raw/su2-compose/2026-07-15-n300-device0-candidate-54b91b-experiment-1/performance.md)

Each package retains the candidate binary and SHA-256, source and environment
provenance, device-health snapshots, raw reports, standard output and error,
and an `artifacts.sha256` inventory. The conformance package also retains the
interrupted performance attempt instead of concealing it.

## Next gate

Before a future Level 2 claim, explicitly select one candidate identity and
freeze a new or recovered preregistration. Then collect three designated,
independent cold-start sessions under that identical contract and run the
deterministic qualifier. Individual reports remain `stable_benchmark=false`.
