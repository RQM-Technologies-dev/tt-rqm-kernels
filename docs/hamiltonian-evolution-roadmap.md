# SU2HamiltonianBench Roadmap

**RQM runs fused time-ordered SU(2) evolution for two-level Hamiltonian
simulation on Tenstorrent Wormhole.** H1 uses CPU-lowered FP32 rotors and phase
pairs. Its protected v3 aggregate is Claim Level 2 for stable fused-only
one-device performance. H2 is the next technical phase; no H2 hardware result
exists yet.

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

H2A is the active implementation milestone:

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

The intended per-core decomposition assigns contiguous flattened coefficient
ranges. The reader moves coefficient and `dt` tiles into circular buffers; the
compute kernel forms squared magnitude, masks exact zero, evaluates square
root/reciprocal and sine/cosine, and produces six component planes; the writer
stores them deterministically. Intermediate precision is FP32. The detailed
pinned-API classification is in the
[experimental H2A scaffold](../experimental/tt_metalium_hamiltonian_lowering/README.md).

### First hardware claim gate

The committed preregistration targets Claim Level 0 conformance only. It
requires one Wormhole device, device identity, pinned TT-Metal and source
commits, candidate binary hash, compiler/runtime provenance, deterministic
serialized input hashes, whole-output validation, and zero failing or
nonfinite values. No designated result may be discarded or replaced.

Until real execution satisfies that contract, H2A remains pre-hardware with
`stable_benchmark=false` and `performance_eligible=false`. It inherits no H1
claim and makes no speedup, stability, full-H2, bandwidth, energy, dual-device,
or endorsement claim.

## H2B: future resident lowering plus H1 composition

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

H2B begins only after H2A correctness is established and H1 compatibility is
revalidated on the selected TT-Metal baseline. It must preserve exact time
order, perform complex128 whole-output comparison, avoid automatic
normalization, and use claim language separate from the historical H1 release.

## Sibling and later work

`EntanglementDynamicsBench` remains a CPU-reference sibling with no hardware
claim. Two-qubit execution, arbitrary quantum-circuit simulation, dual-device
scaling, energy measurement, and general physical-AI applications are outside
H2A/H2B.
