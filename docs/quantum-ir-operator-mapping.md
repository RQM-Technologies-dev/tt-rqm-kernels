# QuantumIR Operator Mapping

This document maps future QuantumIR concepts to the current
`tt-rqm-kernels` operator foundation. It is a design map only; it does not add a
new package or backend.

## Representation Baseline

Current structured quaternion tensors use:

```text
[..., 4] = [real, i, j, k]
```

QuantumIR should reuse this convention when representing SU(2)-like rotations
as quaternion or rotor values. The tensor remains an ordinary floating-point
tensor.

## Initial Mappings

| QuantumIR concept | Current kernel concept | Existing operator path |
| --- | --- | --- |
| SU(2) rotor or single-qubit gate subset | Unit quaternion tensor | `[N, 4]` quaternion convention |
| Gate or rotor composition | Hamilton product | `qmul(a, b)` |
| Unit-rotor stability | Norm and renormalization | `qnorm(q)`, `qnormalize(q)` |
| Adjoint or inverse for a unit rotor | Conjugate or inverse | `qconj(q)`, `qinverse(q)` |
| Bloch/vector-style rotation | Rotor-vector rotation | `qrotate_vector(rotor, vector)` |
| Phase/orientation update | Phase tracking state | `phase_ops` utilities |

The first implementation milestone should not expand this table into a full
quantum-circuit system. It should prove one small mapping, validate it against a
conventional complex reference, and keep the result readable.

## Single-Qubit SU(2) Gate Target

The intended first mapping is:

```text
2x2 complex SU(2) matrix
-> unit quaternion rotor [real, i, j, k]
-> batched structured tensor operation
-> comparison against the original matrix reference
```

The reference should use standard complex matrix multiplication as the check.
The quaternion path should use existing operators such as `qmul`,
`qnormalize`, `qconj`, and `qinverse` where applicable.

## Verified Quaternion-to-SU(2) Conformance

The focused conformance tests use the canonical `rqm-core` unit-quaternion
mapping, written for this repository's `[real, i, j, k] = [w, x, y, z]` lane
order:

```text
U([w, x, y, z]) = [[w - i z, -y - i x],
                   [y - i x,  w + i z]]
```

The mapping preserves the Hamilton-product order used here:

```text
U(qmul(a, b)) = U(a) @ U(b)
```

For unit quaternions, conjugation is the SU(2) adjoint and inverse is the
matrix inverse:

```text
U(qconj(q)) = U(q)†
U(qinverse(q)) = inverse(U(q))
```

This gives `qmul` an independent complex-matrix correctness oracle while
keeping the production representation as ordinary real `[N, 4]` tensors. The
test oracle is reproduced under the test tree and does not add `rqm-core` or
`rqm-entanglement` as runtime dependencies.

The scope is deliberately limited to single-quaternion/SU(2) composition and a
local Kronecker-product composition check. It does not implement general
two-qubit simulation, entangled states, nonlocal gates, or entanglement
analysis.

This target is useful because it exercises:

- four-lane structured values
- cross-lane multiply/add/sign patterns
- unit-norm stability
- inverse/adjoint consistency
- a bridge from conventional quantum notation to ordinary floating-point
  tensors

It does not claim that all quantum workloads reduce to this representation.

## Future Mappings Requiring Separate Design

These topics need separate contracts and validation before code is added:

- controlled rotations and small batched circuit patterns
- Hamiltonian evolution kernels
- batched spin-system workloads
- tensor-network or factored-state representations
- spectral-anchor or coherence diagnostics
- TT-NN or TT-MLIR lowering of any fused QuantumIR operator

Each future mapping should define:

- input and output tensor shapes
- dtype expectations
- reference implementation
- numerical tolerances
- StructuredBench report fields
- backend labels: CPU, simulator, emulation, or hardware

## Non-Goals

This mapping does not:

- define a complete quantum programming language
- claim broad classical simulation efficiency
- replace OpenQASM, Qiskit, TT-NN, TT-MLIR, or TT-Metalium
- request Tenstorrent hardware changes
- create hardware-performance claims from CPU, simulator, or emulation output
