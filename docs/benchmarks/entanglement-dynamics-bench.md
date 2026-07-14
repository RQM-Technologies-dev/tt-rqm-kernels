# EntanglementDynamicsBench Reference Foundation

`EntanglementDynamicsBench` is the first stage of the sibling
`TwoQubitHamiltonianBench` family. The current implementation is a CPU-only
reference foundation: it defines the math, public tensor contracts, independent
oracles, diagnostics, and preregistered correctness cases needed before any
Tenstorrent kernel is designed.

It has no hardware evidence, performance cases, throughput definition, claim
level, or stability claim.

## Contract

Hamiltonians are finite real coefficients with shape `[B,K,4,4]` over ordered
Pauli axes `[I,X,Y,Z] x [I,X,Y,Z]`:

```text
H = sum(C[p,q] * sigma[p] tensor sigma[q])
```

States use real lanes shaped `[B,4,2]`, with basis order
`|00>, |01>, |10>, |11>` and final lanes `[real,imag]`. `[B,8]` is only a
flattened storage view. It is not a second public input contract.

CPU lowering produces `[B,K,4,4,2]` operator lanes. Ordered application uses
step zero first and step `K-1` last:

```text
U[K-1] ... U[0] |psi>
```

Primary outputs are never normalized. Norm drift and physical diagnostics are
computed from the actual result.

## Reference API

The `tt_rqm_kernels.hamiltonian` package exposes:

- `two_qubit_hamiltonian_matrix`
- `lower_two_qubit_hamiltonian`
- `compose_two_qubit_state`
- `evolve_two_qubit_state_reference`
- `apply_local_rotor_pair`
- `two_qubit_state_diagnostics`
- `compare_two_qubit_states`

The real-lane path is checked against an independent complex128
matrix-exponential oracle. Diagnostics record state norm, density trace and
Hermiticity, reduced-state spectra and purity, concurrence, reduced von Neumann
entropy, nonfinite values, and direct and global-phase-aligned oracle error.

## Scientific boundary

The bridge is:

```text
qmul -> local SU(2) -> U_A tensor U_B -> joint-state evolution
     -> nonlocal Hamiltonians -> entanglement metrics
```

Local U(2) operations preserve pure-state entanglement. Nonlocal interaction
terms such as XX, YY, ZZ, and Heisenberg couplings can generate it. H1 remains
a two-level `SU2ComposeBench`; this sibling family does not relabel or extend
its hardware evidence.

The implementation does not add `rqm-entanglement` as a runtime dependency.

## Preregistered checks

The [versioned preregistration](../../benchmarks/manifests/entanglement-dynamics-preregistration.json)
fixes seed 0, `dt=0.05`, `hbar=1`, product and Bell anchors, local-unitary
invariance, XX/YY/ZZ/Heisenberg interactions, reversed noncommuting order,
time-dependent chains, and Float32 drift. Float64/oracle agreement uses
`1e-11`; a future Float32 candidate boundary is `atol=rtol=1e-4` with zero
nonfinite values.

Validate it without hardware:

```bash
python scripts/validate_entanglement_dynamics_preregistration.py
python -m pytest -q tests/test_two_qubit_hamiltonian.py tests/test_entanglement_benchmark.py
```

No hardware release manifest, raw session, plot, TT-Metalium source, or
performance result exists for this benchmark.
