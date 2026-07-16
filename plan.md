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
- Active implementation milestone: H2A device-side two-level Hamiltonian
  coefficient lowering foundation. Hardware execution is not yet implemented.
- Future integration: H2B device-resident H2A lowering directly feeding the
  protected fused H1 composition path.
<!-- repository-claims:end -->

Neither Level 2 release is a CPU or application acceleration claim. The
repository has no stable fused/unfused comparison, measured-bandwidth, energy,
dual-device, full device-side coefficient-lowering, or Tenstorrent-endorsement
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

## Active implementation milestone: H2A

H2A accepts FP32 Hamiltonian coefficients `[B,K,4]` in `[h0,hx,hy,hz]`
order and scalar or broadcastable `[B,K]` FP32 `dt`. It produces FP32 rotors
`[B,K,4]` and phases `[B,K,2]` for the existing H1 boundary.

The current H2A scope is an implementation-ready CPU reference benchmark,
independent analytical and complex128 oracles, a fail-closed external candidate
protocol, a pinned-API design audit, and a pre-hardware Claim Level 0
preregistration. It contains no TT-Metal candidate, hardware report,
performance-eligible result, or release manifest.

The first hardware milestone is conformance only: one Wormhole device, pinned
candidate/source/runtime provenance, deterministic serialized input hashes,
whole-output validation, zero failing and nonfinite values, and
`stable_benchmark=false`.

## Future integration: H2B

H2B is device-resident H2A lowering directly feeding fused H1 ordered
composition without a host round-trip for intermediate rotors or phases. Its
input is coefficients plus `dt`; its output is final `[B,4]` rotors and
`[B,2]` phases.

H2B waits for H2A correctness, revalidation of H1 compatibility on the chosen
TT-Metal baseline, exact time-order tests, and complex128 whole-output checks.
It will receive new claim language and cannot reuse historical H1 status.

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
- [Operator contracts](docs/operator-contracts.md)
- [Benchmark claim policy](docs/benchmarks/claim-policy.md)
- [Documentation index](docs/index.md)
