# SU2HamiltonianBench Roadmap

**RQM runs fused time-ordered SU(2) evolution for two-level Hamiltonian
simulation on Tenstorrent Wormhole.** H1 uses CPU-lowered FP32 rotors and phase
pairs. Its protected v3 aggregate is Claim Level 2 for stable fused-only
one-device performance. H2 is the next technical phase. H2A Claim Level 0
silicon conformance is established from one separately designated N300
device-0 session; it is not a performance or stability result.

## Protected baselines

Stage A scalar qmul remains the immutable correctness baseline. Persistent
Stage B qmul remains the protected Claim Level 2 qmul release. H1 v3 remains
the protected fused-composition Claim Level 2 release. Their source commits,
candidates, manifests, and raw evidence must not be rewritten by H2 work.

A port to a newer TT-Metal commit requires fresh whole-output correctness and
profiler validation. It cannot inherit a stable label because its semantics
match an older candidate.

## H1: completed fused-composition baseline

```text
rotors: [B, K, 4] Float32, [w, x, y, z]
phases: [B, K, 2] Float32, [real, imag]

total_rotor = rotor[K-1] * ... * rotor[0]
total_phase = phase[K-1] * ... * phase[0]
```

The primary variant never normalizes automatically. The historical v2
fused/unfused campaign is retained and non-qualifying. The later v3 campaign
qualified three designated sessions under a fused-only contract. H1 therefore
does not claim stable fused/unfused comparison performance or acceleration.

## H2A: device-side coefficient lowering

H2A is the completed coefficient-lowering conformance milestone:

```text
hamiltonians [B,K,4] + dt
        -> H2A coefficient lowering
        -> rotors [B,K,4] + phases [B,K,2]
```

### Logical input and output contract

```text
hamiltonians: [B, K, 4] FP32
lanes:        [h0, hx, hy, hz]

dt: scalar FP32 or broadcastable [B,K] FP32
hbar: finite positive scalar

rotors: [B, K, 4] FP32
lanes:  [w, x, y, z]

phases: [B, K, 2] FP32
lanes:  [real, imag]
```

For each piecewise-constant Hamiltonian,

```text
H = h0 I + hx sigma_x + hy sigma_y + hz sigma_z
r = sqrt(hx^2 + hy^2 + hz^2)
theta = r * dt / hbar
alpha = h0 * dt / hbar

rotor = [cos(theta),
         sin(theta) * hx/r,
         sin(theta) * hy/r,
         sin(theta) * hz/r]

phase = [cos(alpha), -sin(alpha)]
```

At exactly `r = 0`, the rotor is exactly `[1,0,0,0]`; the phase remains
`[cos(alpha),-sin(alpha)]`. Tiny nonzero norms follow the nonzero equation.
There is no epsilon branch, hidden clamping, or automatic normalization.

The signs match the repository convention

```text
U([w,x,y,z]) = [[w - i z, -y - i x],
                [y - i x,  w + i z]]
```

multiplied by the complex phase `phase.real + i*phase.imag`. With the negative
sine in the phase and the rotor convention above, this reconstructs
`exp(-i H dt / hbar)`. Later H1 multiplication remains
`step[K-1] * ... * step[0]`.

### Validation and exceptional inputs

- The public H2A output is FP32, compared against an independent Float64
  analytical reference and complex128 `matrix_exp` reconstruction.
- Zero, axis-aligned, tiny-norm, deterministic random, varying-`dt`,
  large-angle, and mixed zero/nonzero cases are required.
- Large finite angles are accepted; approximation accuracy is measured rather
  than hidden by range clamping.
- Wrong shape, integer or complex dtype, incompatible `dt`, NaN, infinity, and
  nonpositive or nonfinite `hbar` fail before execution.
- Every returned lane is validated; malformed, truncated, reordered, or
  nonfinite candidate output fails closed.
- Rotor and phase norm drift are diagnostics. Outputs are never normalized to
  conceal it.

### Logical-to-device layout boundary

The logical API remains ordinary `[B,K,*]` tensors and requires no quaternion
datatype. A TT-Metal candidate may pack `h0`, `hx`, `hy`, `hz`, and broadcast
`dt` into component-planar FP32 tiles, padding the flattened `B*K` dimension to
tile boundaries. It must unpad and restore deterministic row-major `[B,K,*]`
ordering at the protocol boundary.

The implemented candidates use one Tensix core. Their reader moves six
component planes into circular buffers; compute forms squared magnitude,
performs exact-zero selection before reciprocal, evaluates native Wormhole
square root and trigonometry, and produces six output planes; the writer stores
them deterministically. Intermediate precision is FP32. Pinned `where_tile` is
not valid for FP32, so the candidate uses a custom lane-wise SFPI selection.
The detailed architecture is in the
[experimental H2A candidate](../experimental/tt_metalium_hamiltonian_lowering/README.md).

The original candidate's frozen large-angle case produced `9.36e-04` maximum
rotor error and `4.34e-04` maximum phase error. Diagnostics localized the loss
to ordinary one-value FP32 angle multiplication before trigonometry; `r²` was
exact for the case, while pinned sine/cosine already used Cody–Waite reduction.

The distinct [compensated Candidate B](../experimental/tt_metalium_hamiltonian_lowering_compensated/README.md)
uses Dekker split products and carries high/low `theta` and `alpha` through a
split-`2π` device reduction. SFPMAD was available but rejected after its
hardware residual proved quantized. No corrected magnitude or square root was
needed. Candidate B passed all nine frozen cases and one retained
non-designated pilot; the large case had zero failing values, `1.17e-04`
maximum rotor error, and `4.73e-08` maximum phase error. This is still not
Claim Level 0 evidence by itself.

The candidate is now bound to implementation commit `225cb213…` and source
bundle `519b2b9f…`. Two isolated clean builds produced byte-identical binary
`b12063fd…`; the clean primitive, axis, mixed, large-angle, and nine-case runs
passed, and every nine-case output checksum matched the retained dirty-tree
pilot. These are explicitly non-designated development reproductions.

### First hardware claim gate

The committed preregistration targets Claim Level 0 conformance only. It
requires one Wormhole device, device identity, pinned TT-Metal and source
commits, candidate binary hash, compiler/runtime provenance, deterministic
serialized input hashes, whole-output validation, and zero failing or
nonfinite values. No designated result may be discarded or replaced.

The [frozen designated manifest](../benchmarks/manifests/hamiltonian-lowering-h2a-designated-conformance.json),
[serialized inputs](../benchmarks/inputs/hamiltonian-lowering-h2a-designated-conformance),
collector, qualifier, and [runbook](hamiltonian-lowering-h2a-designated-runbook.md)
were frozen before collection. The retained designated session then passed all
nine cases with one attempt per case and no retries or replacements. The
[separate public report](benchmarks/hamiltonian-lowering-h2a.md) and
[release manifest](../benchmarks/manifests/wormhole-hamiltonian-lowering.json)
establish Claim Level 0 silicon conformance.

The release remains `stable_benchmark=false` and
`performance_eligible=false`. It inherits no H1 claim and makes no speedup,
stability, full-H2, bandwidth, energy, dual-device, or endorsement claim.

## H2B: resident lowering plus H1 composition; first pilot failed

H2B is precisely:

```text
input:
  coefficients [B,K,4]
  dt scalar or [B,K]

device-resident H2A lowering
  -> existing fused H1 ordered composition
  -> no host round-trip of intermediate rotors or phases

output:
  final rotor [B,4]
  final phase [B,2]
```

H2B now has a direct CPU API, deterministic `HamiltonianEvolutionBench`
semantic cases, complex128 whole-chain comparison, a fail-closed external
protocol, and a real two-program TT-Metal candidate source. The candidate
creates Wormhole device 0 once, writes the six-plane coefficient input once,
runs compensated H2A into a device-DRAM intermediate, runs protected fused H1
against that exact buffer, reads only final rotor/phase planes, and closes the
device once. The first non-designated N300 pilot is retained and did not pass.
All 20 frozen invocations stopped before device execution because the launcher
did not set the separately required TT-Metal runtime root. This is an
`environment` failure, not a numerical result; no case was retried or replaced.

The H2A and intermediate layouts both use
`(step*6+lane)*component_tiles+batch_tile`. Input lanes are
`[h0,hx,hy,hz,dt,inverse_hbar]`; H2A output/H1 input lanes are
`[w,x,y,z,phase_real,phase_imag]`. There is no intermediate host read, host
repacking, or host write. This is a device-resident pipeline whose H1 stage is
fused, not one fused kernel. H2B remains `stable_benchmark=false`,
`performance_eligible=false`, and `claim_level=null`, and inherits no H1 or
H2A evidence.

The next hardware action, if pursued, requires a new versioned non-designated
contract that explicitly binds the runtime root. Session 1 remains immutable.

## Sibling and later work

`EntanglementDynamicsBench` remains a CPU-reference sibling with no hardware
claim. Two-qubit execution, arbitrary quantum-circuit simulation, dual-device
scaling, energy measurement, and general physical-AI applications are outside
H2A/H2B.
