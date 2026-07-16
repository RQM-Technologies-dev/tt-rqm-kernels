# H2B Device-Resident Hamiltonian Evolution Foundation

HamiltonianEvolutionBench H2B now has a CPU/reference foundation, a fail-closed
external protocol, and a TT-Metal candidate source. Hardware has not yet been
run. H2B has no claim level and remains `stable_benchmark=false` and
`performance_eligible=false`; `claim_level=null`.

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

Current status: **CPU/reference foundation implemented; TT-Metal candidate
source present; hardware not yet run**.

This foundation is not stable, performance-eligible, accelerated, or Claim
Level 0 or higher. It does not inherit H1 stability or H2A conformance. It does
not claim a single fused kernel, avoidance of device DRAM, or Tenstorrent
endorsement.
