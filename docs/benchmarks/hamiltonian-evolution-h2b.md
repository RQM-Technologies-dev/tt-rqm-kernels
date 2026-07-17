# H2B Device-Resident Hamiltonian Evolution Pilot

HamiltonianEvolutionBench H2B has a CPU/reference foundation, fail-closed
external protocol, TT-Metal candidate source, and two retained Contract-v1
non-designated N300 sessions. Session 1 remains immutable. Session 2 passed
the corrected launcher preflight but did not pass; the first evidenced failing
layer is `runtime`, with dispatch/mailbox synchronization during device
initialization. H2B has no claim
level and remains `stable_benchmark=false`, `performance_eligible=false`, and
`claim_level=null`.

## Contract

```text
hamiltonians [B,K,4] FP32 [h0,hx,hy,hz]
dt scalar FP32 or [B,K] FP32
  -> final_rotor [B,4] FP32 [w,x,y,z]
  -> final_phase [B,2] FP32 [real,imag]
```

The public `evolve_two_level_hamiltonian` API uses the validated H2A lowering
contract and exact H1 `step[K-1] * ... * step[0]` composition. It never
normalizes output. The independent whole-chain oracle is complex128
`compose_hamiltonian_matrices`, not the rotor/phase path itself.

Deterministic cases cover identity K=1, zero-vector phase accumulation,
x/y/z axes, reversed noncommuting x/y order, mixed zero/nonzero steps, tiny
norms, varying `dt`, random inputs, K=512 drift, and large angles. The family
includes K=1, 2, 8, 32, 128, and 512 and reports final rotor/phase absolute and
relative errors, failing and nonfinite counts, norm drift, direct and
global-phase-aware complex128 matrix errors, and an output checksum.

## Large-angle diagnosis and bounded domain

The retained large-angle error originates primarily in uncompensated FP32
angle-product formation, not H1 composition. In the deterministic stage
diagnostic, uncompensated lowering reached `7.835e-4` rotor error and
`1.040e-3` per-step matrix error. Protected compensated-H2A-equivalent
lowering reduced those to `1.169e-4` and `1.261e-4`, while Float32 H1
composition of exact steps contributed only `1.207e-7` final matrix error.

The wider mixed-direction sweep first failed at rotor angle
`1539.3794236964986` after passing through `1536.238807605409`. The frozen
conformance contract therefore uses the conservative physical API bounds
`abs(theta) <= 1024` and `abs(alpha) <= 8192` radians per logical step, for
both signs. Inputs outside this implementation domain remain mathematically
valid but are unsupported by the H2B conformance contract and fail closed.
`large_angle_short_chain` remains retained as an out-of-domain stress
diagnostic rather than a conformance gate.

## Frozen pilot contract

The preregistration was frozen before candidate execution with 20 cases in
this exact order: `identity_k1`, `zero_vector_phase_chain`, `axis_x`, `axis_y`,
`axis_z`, `noncommuting_xy`, `noncommuting_yx`, `mixed_zero_nonzero`,
`tiny_norms`, `varying_dt`, `random_finite`, `long_chain`,
`boundary_rotor_positive`, `boundary_rotor_negative`,
`boundary_phase_positive`, `boundary_phase_negative`, `boundary_combined`,
`boundary_noncommuting_xy`, `boundary_noncommuting_yx`, and
`large_angle_short_chain`.

The contract binds one attempt, zero retries or replacements, `atol=rtol=1e-4`,
zero failing and nonfinite values, final matrix maximum absolute error
`2e-4`, exact inputs, source and binary identities, TT-Metal commit, device 0,
two programs, and the no-intermediate-transfer lifecycle.

## External protocol

The versioned protocol is `tt-rqm-external-hamiltonian-evolution.v1`, with
metrics schema `tt-rqm-external-hamiltonian-evolution-metrics.v1` and report
schema `tt-rqm-hamiltonian-evolution-candidate-report.v1`. It serializes
`hamiltonians.bin`, `dt.bin`, and `manifest.json`, and requires exact
`final_rotors.bin`, `final_phases.bin`, and `metrics.json` outputs. Validation
fails closed on shapes, byte counts, lane order, row-major ordering, nonfinite
data, exit status, lifecycle, provenance, transfers, and claim escalation.

```bash
python scripts/validate_hamiltonian_evolution_candidate.py \
  --command "python scripts/hamiltonian_evolution_external_reference.py"
```

## TT-Metal architecture

The candidate is two programs in one Wormhole device-0 session, not one fused
kernel. Compensated H2A uses one Tensix core and writes six step-major planes
to an intermediate `MeshBuffer` in device DRAM. That same buffer becomes the
input to the unchanged protected fused H1 reader. H1 splits trajectory tiles
across available cores and retains six accumulator lanes in L1.

```text
page = (step * 6 + lane) * component_tiles + batch_tile
component_tiles = ceil(B / 1024)
```

Input lanes are `[h0,hx,hy,hz,dt,inverse_hbar]`; intermediate lanes are
`[w,x,y,z,phase_real,phase_imag]`. Scalar `dt` is expanded during packing.
There is one H2D input write, no intermediate D2H read, no host repacking, no
intermediate H2D write, and one final D2H read. Metadata requires
`program_count=2`, one device create/close, zero intermediate transfer counts,
and `host_round_trip_count=0`.

## Current status and nonclaims

Current status: **Contract-v1 Session 2 is retained and did not pass. The first
evidenced failing layer is `runtime`. No H2B hardware claim exists.**

All 20 frozen cases were invoked once in order without retry or replacement.
The standalone and collector preflights propagated both runtime roots and
validated the pinned source, binary, TT-Metal checkout, shared libraries,
fresh cache root, and both N300 entries. Candidate execution then stalled in
runtime/dispatch synchronization; retained logs for 19 cases report active
dispatch cores, failure to complete early exit, and unexpected run-mailbox
value `0x40`. No metrics or final numerical outputs were produced. Both N300
entries remained healthy and visible after collection. The independent
Session 2 qualifier reports `package_valid=true` and `pilot_passed=false`.

This foundation is not stable, performance-eligible, accelerated, or Claim
Level 0 or higher. It does not inherit H1 stability or H2A conformance. It does
not claim a single fused kernel, avoidance of device DRAM, or Tenstorrent
endorsement.

No designated Claim Level 0 contract was prepared because Session 2 did not
pass. Session 1 and Session 2 remain separately retained; neither may be
retried, replaced, or overwritten.
