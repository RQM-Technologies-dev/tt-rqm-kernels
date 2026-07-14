# SU2HamiltonianBench Roadmap

**RQM runs fused time-ordered SU(2) evolution for two-level Hamiltonian
simulation on Tenstorrent Wormhole.** H1 lowers piecewise-constant two-level
Hamiltonian coefficients into FP32 rotors and phase pairs on the CPU, and
Wormhole performs their ordered composition. H2 will address device-side
coefficient lowering. H1 is a real pipeline stage, not the complete device-side
pipeline.

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

## Sibling Family: TwoQubitHamiltonianBench

`SU2HamiltonianBench` remains unchanged for H1 and H2. The sibling
`TwoQubitHamiltonianBench` begins with `EntanglementDynamicsBench`, whose first
milestone is an implemented CPU reference rather than a hardware stage.

```text
qmul -> local SU(2) -> U_A tensor U_B -> joint-state evolution
     -> nonlocal Hamiltonians -> entanglement metrics
```

The reference accepts real `[B,K,4,4]` Pauli-product coefficients and
`[B,4,2]` state lanes, applies exact time order without normalization, and
compares against a complex128 matrix-exponential oracle. Local U(2) operations
preserve entanglement; nonlocal interaction terms can generate it.

The foundation has no TT-Metalium code, hardware evidence, performance cases,
claim level, or stability claim. It does not use `rqm-entanglement` as a runtime
dependency. See the
[reference contract](benchmarks/entanglement-dynamics-bench.md).

## Later Work

`RigidBodyHamiltonianBench` is a separate physical-AI follow-on. It does not
share H1 claims and will not begin until the SU(2) benchmark is stable.

## Non-goals

- arbitrary quantum-circuit simulation;
- quantum-hardware replacement claims;
- CPU speedup, energy, or bandwidth claims without matched evidence;
- dual-device results during H1;
- automatic normalization that hides accumulated numerical error.
