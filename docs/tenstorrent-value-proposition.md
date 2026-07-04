# Tenstorrent Value Proposition

`tt-rqm-kernels` gives Tenstorrent a compact, structured workload family between
scalar elementwise operators and large matmul-heavy applications.

The public value is not quantum-hardware replacement. The value is a reusable
operator and benchmark pack for structured tensor states represented inside
ordinary floating-point tensors.

## What RQM Technologies Can Contribute

RQM Technologies can help the Tenstorrent ecosystem with:

- structured non-LLM kernels
- signal processing kernels
- phase, orientation, and wave-state operators
- physical AI pose and orientation stream workloads
- scientific computing benchmark patterns
- compact four-lane tensor operators that stress cross-lane dependencies
- benchmark reports that distinguish CPU, simulator, emulation, and hardware
  runs
- a future QuantumIR for Classical AI Compute layer for selected
  quantum-mechanics and AI augmentation workloads on classical AI accelerators

## Why The qmul Wedge Matters

The first functional target is `qmul` over `[N, 4]` floating-point tensors:

```text
[N, 4] = [real, i, j, k]
```

It is small enough to validate with CPU/PyTorch and scalar references, but it is
structured enough to exercise:

- cross-lane multiply/add/sign dependencies
- fixed four-lane layout
- data movement and register reuse
- fusion opportunities
- arithmetic intensity tradeoffs
- deterministic report comparison across backend modes

This makes `qmul` useful as a lower-stack benchmark even before larger
application workloads are introduced.

## Classical Accelerator Framing

The future QuantumIR layer should be read as a classical/AI accelerator front
end for selected quantum-mechanics workloads, not as a quantum-hardware
proposal.

Longer term, QuantumIR could lower selected SU(2) rotations, unitary
composition, Hamiltonian evolution, phase/coherence updates, and AI
augmentation workloads into the same structured quaternion, rotor, phase, and
tensor operators used by StructuredBench.

## Guardrails

This project does not claim:

- arbitrary quantum computation is efficiently classically simulable
- RQM replaces quantum hardware
- Tenstorrent endorses RQM theory
- simulator or emulation output is hardware performance
- CPU/PyTorch reports predict Tenstorrent hardware performance

This project does not ask for:

- native quaternion hardware
- a new Tenstorrent chip feature
- defense-first positioning

The near-term engineering goal is simpler: make structured kernels easy to run,
validate, compare, and extend on classical Tenstorrent accelerator paths.
