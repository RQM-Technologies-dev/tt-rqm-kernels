# tt-rqm-kernels Plan

This plan separates protected results from active implementation and future
integration. Machine-readable release manifests and their deterministic
validators remain the source of truth.

## Current claim status

<!-- repository-claims:start -->
- qmul: Claim Level 2 stable one-device performance; aggregate
  `stable_benchmark=true` from three qualified sessions.
- SU2ComposeBench H1: Claim Level 2 stable one-device **fused-only**
  performance; aggregate `stable_benchmark=true` from three qualified v3
  sessions.
- Every individual qmul and H1 source-session report remains
  `stable_benchmark=false`.
- The historical H1 v2 fused/unfused campaign is retained and non-qualifying;
  it is not the current release.
- H2A device-side coefficient lowering: Claim Level 0 silicon conformance from
  one designated N300 device-0 session; `stable_benchmark=false` and
  `performance_eligible=false`.
- H2B first non-designated N300 pilot: retained and did not pass; the failure
  is classified as environment. All 20 frozen cases were attempted once
  without retry or replacement. No H2B hardware claim exists;
  `stable_benchmark=false`, `performance_eligible=false`, `claim_level=null`.
<!-- repository-claims:end -->

Neither Level 2 release is a CPU or application acceleration claim. The
repository has no stable fused/unfused comparison, measured-bandwidth, energy,
dual-device, complete device-resident H2 pipeline, or Tenstorrent-endorsement
claim.

## Protected completed baselines

### qmul

- Stage A scalar RISC-V qmul is the immutable correctness baseline.
- Stage B persistent multicore Tensix/SFPU qmul is the protected Claim Level 2
  hardware release.
- A port to another TT-Metal commit requires fresh whole-output correctness and
  profiler validation. Matching semantics alone does not transfer the stable
  label.

### H1 fused composition

- CPU lowering produces FP32 rotors and phase pairs.
- The protected H1 v3 candidate composes them in exact time order on one
  Wormhole device.
- Its Level 2 claim is fused-only. Unfused values remain diagnostics and do not
  establish stable fused/unfused comparison performance.
- New work must not alter H1 raw evidence, silently replace its candidate, or
  inherit its stable label.

The v2 H1 campaign remains a complete historical record. All three designated
sessions were retained, but the frozen fused, unfused, and paired-ratio gates
did not qualify. The later v3 campaign used a separately frozen fused-only
contract and passed with three designated sessions.

## Protected H2A conformance milestone

H2A accepts FP32 Hamiltonian coefficients `[B,K,4]` in `[h0,hx,hy,hz]`
order and scalar or broadcastable `[B,K]` FP32 `dt`. It produces FP32 rotors
`[B,K,4]` and phases `[B,K,2]` for the existing H1 boundary.

The current H2A scope now includes the CPU reference benchmark, independent
analytical and complex128 oracles, fail-closed external protocol, pinned API
audit, a real single-core TT-Metalium candidate, and non-designated pilot
collection/validation machinery. The original candidate's large-angle probe
failed because one-value FP32 angle formation discarded product residuals
before trigonometric reduction. A distinct Candidate B uses split TwoProduct
and split-`2π` device reduction; all nine frozen cases and its one retained
non-designated pilot pass. Neither the original blocker nor the pilot is a
claim level, performance result, or release. The clean implementation commit is
`225cb213…`; its 26-file source bundle is `519b2b9f…`, and two isolated builds
produced the same `b12063fd…` binary. Clean-tree N300 output checksums matched
the retained pilot exactly.

The designated Claim Level 0 contract, exact serialized inputs, fail-closed
collector, and offline qualifier were frozen before collection. One designated
N300 device-0 session then passed all nine cases with pinned
candidate/source/runtime provenance, whole-output validation, zero failing and
nonfinite values, one attempt per case, and no retries or replacements. The
separate public Level 0 release remains `stable_benchmark=false` and
performance-ineligible; it does not confer H2B evidence or inherit H1 claims.

## Active integration: H2B failed non-designated pilot

H2B is device-resident H2A lowering directly feeding fused H1 ordered
composition without a host round-trip for intermediate rotors or phases. Its
input is coefficients plus `dt`; its output is final `[B,4]` rotors and
`[B,2]` phases.

The reference API, exact-order tests, complex128 whole-output oracle,
deterministic benchmark family, external protocol, and two-program candidate
source now exist. The candidate creates device 0 once, runs compensated H2A
and protected fused H1 against one device-DRAM intermediate, reads only final
rotor/phase output, and closes once. The first frozen 20-case N300 pilot is
retained, but every invocation stopped before device execution because the
launcher omitted the separately required TT-Metal runtime-root variable. The
failure is classified as `environment`; no retry or replacement occurred.
H2B requires a newly versioned non-designated contract before any future run
and cannot reuse historical H1 or H2A status.

## Deferred work

- qmul upstream placement remains pending maintainer guidance in
  [tenstorrent/tt-metal#49887](https://github.com/tenstorrent/tt-metal/issues/49887).
  No upstream port begins while it is unanswered.
- TT-NN and TT-MLIR integration remain deferred; no fake bindings or native
  quaternion datatype are planned.
- CPU matched-scope timing, energy measurement, dual-device scaling, and
  two-qubit hardware execution require separate methodologies.
- `RigidBodyHamiltonianBench` and broader physical-AI studies remain separate
  follow-ons and inherit no qmul or H1 evidence.

## Primary references

- [SU2ComposeBench report](docs/benchmarks/su2-compose-bench.md)
- [Wormhole qmul report](docs/benchmarks/wormhole-qmul.md)
- [H2A/H2B roadmap and contract](docs/hamiltonian-evolution-roadmap.md)
- [H2A Claim Level 0 report](docs/benchmarks/hamiltonian-lowering-h2a.md)
- [H2B foundation](docs/benchmarks/hamiltonian-evolution-h2b.md)
- [Operator contracts](docs/operator-contracts.md)
- [Benchmark claim policy](docs/benchmarks/claim-policy.md)
- [Documentation index](docs/index.md)
