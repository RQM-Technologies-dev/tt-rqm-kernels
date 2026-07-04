# QuantumIR Direction

QuantumIR is a long-term direction for `tt-rqm-kernels`: a domain-facing layer
that lowers selected quantum-mechanics workloads into the structured quaternion,
rotor, phase, and tensor kernels already defined by this repository.

This is not a replacement for the current lower-stack work. The immediate
engineering path remains:

```text
CPU/PyTorch reference kernels
-> scalar correctness checks
-> StructuredBench reports
-> TT-Lang simulator qmul
-> tt-emule validation of a TT-Metalium qmul candidate
-> future Tenstorrent Cloud or hardware report
```

QuantumIR is the proposed layer above that path. It should preserve useful
structure from quantum-mechanics models before lowering to ordinary
floating-point tensor kernels.

## Thesis

Many quantum-mechanics workloads contain structured operations such as SU(2)
rotations, phase updates, unitary composition, Hamiltonian evolution, and
factored state updates. Those structures can often be represented with compact
floating-point tensor conventions before they are lowered into accelerator
kernels.

For this repo, the starting convention is still:

```text
[N, 4] = [real, i, j, k]
```

The value is an ordinary real-valued tensor. The structure comes from the
operator contract and validation path.

The long-term stack is:

```text
OpenQASM / Qiskit / RQM DSL
-> QuantumIR / RQM-IR
-> SU(2), quaternion, rotor, phase, spectral, and tensor lowering
-> StructuredBench and tt-rqm-kernels operator contracts
-> TT-MLIR / TT-NN / TT-Metalium / TT-Lang / tt-emule paths
-> Tenstorrent hardware when a real hardware environment is available
```

This positions RQM Technologies as a structured-kernel and quantum-dynamics
front end for selected workloads on classical accelerators. It does not require
a native quaternion datatype, a new Tenstorrent silicon feature, or any change
to the current Tenstorrent programming model.

## First Concrete Target

The first QuantumIR target should be a single-qubit SU(2) gate represented as a
unit quaternion rotor.

The validation path should be conventional and testable:

```text
single-qubit SU(2) matrix reference
-> equivalent unit quaternion rotor representation
-> batched application using existing qmul/qnormalize-style kernels
-> numerical comparison against the 2x2 complex matrix reference
```

This first target is intentionally small. It connects quantum-mechanics notation
to the existing `[N, 4]` kernel foundation without adding a new backend or
claiming hardware performance.

## Relationship To Existing Work

QuantumIR should lower into the existing library rather than replace it:

- `qmul` remains the first lower-stack kernel wedge.
- StructuredBench remains the report and comparison surface.
- TT-Lang and tt-emule remain simulator/emulation evidence, not hardware
  performance.
- Future TT-Metalium and Tenstorrent Cloud runs remain the path to real
  hardware evidence.
- The LWT/ILWT external contribution path remains separate from this repo.

## Non-Goals

QuantumIR does not claim:

- arbitrary quantum computation is efficiently classically simulable
- RQM replaces quantum hardware
- Tenstorrent endorses this project
- Tenstorrent should add native quaternion hardware
- CPU, simulator, or emulation output is hardware performance
- speculative RQM physics claims as established engineering facts

The near-term standard is engineering evidence: explicit tensor contracts,
reference validation, reproducible reports, and conservative labels for CPU,
simulator, emulation, and hardware runs.
