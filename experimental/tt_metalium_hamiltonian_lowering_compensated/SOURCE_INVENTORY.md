# H2A compensated milestone source inventory

The implementation milestone intentionally contains four classes of files:

- candidate implementation: the original single-core host/reader/writer
  boundary plus the distinct compensated compute and SFPI sources;
- validation machinery: build/run wrappers, pinned-environment audit,
  deterministic source identity, external protocol, pilot collector and
  validators, CI hooks, and focused tests;
- retained development evidence: the immutable original blocker, compensated
  diagnostic comparison, and passing non-designated nine-case N300 pilot;
- public governance surfaces: README, roadmap, benchmark index, operator
  contract, plan, repository status, and cross-document claim validator.

The deterministic candidate source bundle is limited to files under
`experimental/tt_metalium_hamiltonian_lowering` and
`experimental/tt_metalium_hamiltonian_lowering_compensated` whose suffix is
`.cpp`, `.h`, `.py`, or `.txt`, plus the benchmark contract, external candidate
protocol, and source-identity modules used to execute and verify that code.
Relative names and bytes are hashed; absolute paths, timestamps, build outputs,
and caches are excluded. The generated source manifest records every included
path, size, and SHA-256.

Excluded temporary state includes `build/`, alternate build directories,
`__pycache__/`, `.pytest_cache/`, TT-Metal JIT caches, remote scratch packages,
and local shell logs. The audit found no unrelated pre-existing edit in the
milestone worktree.
