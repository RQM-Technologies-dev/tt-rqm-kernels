# Raw benchmark sessions

Timestamped hardware packages live here. Each package retains the command,
candidate hash, environment, device-health snapshots, stdout/stderr, reports,
and a hash-bound session manifest. Canonical artifacts in `reports/` are never
overwritten by this workflow.

`wormhole-qmul-*` packages support the published qmul evidence. The
`su2-compose/` subtree contains the historical Claim Level 1 comparison, the
separate `54b91b…` candidate experiment and profiler attempts, and the three
designated v2 stability-session packages. The v2 packages are retained even
though their deterministic qualification is non-passing; none was replaced or
discarded. See the [qualification artifact](../processed/wormhole-su2-compose-stability-qualification.json)
and [SU2 evidence index](../../docs/benchmarks/index.md).
