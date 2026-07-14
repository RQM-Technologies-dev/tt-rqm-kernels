# SU2HamiltonianBench Roadmap

RQM is building quantum Hamiltonian simulation benchmarks for Tenstorrent.
The first stage, `SU2ComposeBench`, executes the ordered composition portion of
piecewise-constant two-level evolution. It consumes evolution operators lowered
on the CPU; it does not yet lower Hamiltonian coefficients on Wormhole.

## Hamilton Product And Hamiltonian

`qmul` is the Hamilton product of two quaternions. A two-level Hamiltonian is a
Hermitian matrix

```text
H = h0 I + hx sigma_x + hy sigma_y + hz sigma_z.
```

For a step of duration `dt`, its U(2) evolution separates into a scalar phase
and an SU(2) rotor. `SU2ComposeBench` begins after that separation and measures
ordered, noncommuting composition on Wormhole.

## H1: SU2ComposeBench

```text
rotors: [B, K, 4] Float32, [w, x, y, z]
phases: [B, K, 2] Float32, [cos(alpha), -sin(alpha)]

total_rotor = rotor[K-1] * ... * rotor[0]
total_phase = phase[K-1] * ... * phase[0]
```

The primary variant never normalizes the result automatically. The benchmark
compares a fused chain kernel with an unfused repeated-qmul path on the same
device and under the same timing boundary.

## H2: Device-Side Lowering

H2 will accept `[B,K,4]` Hamiltonian coefficients and scalar or `[B,K]` `dt`,
perform norm, axis, sine, cosine, phase, and rotor construction on device, and
then compose the chain. H2 starts only after H1 is stable and the pinned
TT-Metal SFPU transcendental APIs have been audited.

Only H2 may be described as full device-side two-level Hamiltonian evolution
lowering.

## Later Work

`RigidBodyHamiltonianBench` is a separate physical-AI follow-on. It does not
share H1 claims and will not begin until the SU(2) benchmark is stable.

## Non-goals

- arbitrary quantum-circuit simulation;
- quantum-hardware replacement claims;
- CPU speedup, energy, or bandwidth claims without matched evidence;
- dual-device results during H1;
- automatic normalization that hides accumulated numerical error.
