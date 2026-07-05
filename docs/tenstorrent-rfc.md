# RFC: Structured Quaternion and Rotor Kernels for Tenstorrent

## Summary

`tt-rqm-kernels` proposes structured quaternion and rotor kernels represented inside ordinary floating-point tensors.

The core convention is:

```text
[..., 4] = [real, i, j, k]
```

The tensor remains a standard floating-point tensor. The structure comes from the operator contract: kernels interpret the final dimension as a quaternion and apply structured algebra such as Hamilton product, conjugation, normalization, inverse, and rotor-vector rotation.

This RFC is intended for Tenstorrent maintainers and engineers as an early placement and integration discussion.

## Decision Requested

Where should a minimal `[N, 4]` structured `qmul` example live?

Preferred maintainer guidance would be one of:

- keep it as an external community example
- place it as a TT-Metalium programming example
- treat it as a TT-NN custom-op precursor after the lower-stack path is proven
- defer TT-MLIR discussion until a real backend comparison exists
- not a fit for upstream, but useful as a community benchmark

The immediate ask is narrow: placement guidance and, if possible, one real
hardware validation run for the existing StructuredBench `qmul` path.

## What We Are Not Asking For

- no native quaternion hardware
- no new datatype
- no new silicon feature
- no Tenstorrent endorsement of RQM theory
- no defense-first framing
- no hardware-performance claim without real hardware

## Motivation

Quaternion, rotor, phase, and orientation operations appear in several numerical workloads:

- robotics
- graphics
- physical AI
- wireless
- imaging
- simulation
- scientific computing
- signal processing
- defense as a downstream application

These are not LLM-specific operators. They are structured numerical workloads that can still map naturally onto tensor hardware because the data representation is ordinary floating-point tensor data.

The project is intended to demonstrate a non-LLM structured numerical workload for Tenstorrent hardware. The initial goal is not to introduce a new abstraction into the hardware. It is to show that useful structured math can be expressed as small, clear kernels with:

- explicit tensor shape contracts
- CPU/PyTorch reference validation
- deterministic correctness tests
- throughput, latency, numerical-error, and scaling benchmarks
- StructuredBench reports that can later compare CPU/PyTorch, TT-Metalium, and TT-NN backends

## Proposed First Integration

The proposed first integration is a minimal TT-Metalium quaternion multiply example.

### Operator

```text
qmul(a, b) -> out
```

Inputs:

```text
a:   [..., 4]
b:   [..., 4]
```

Output:

```text
out: [..., 4]
```

The final dimension is interpreted as:

```text
[w, x, y, z] = [real, i, j, k]
```

### Operation

The operation is the Hamilton product:

```text
out.w = a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z
out.x = a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y
out.y = a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x
out.z = a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w
```

The first version should prioritize clarity, correctness, and stable benchmarking over peak optimization.

### Validation

Validation should compare TT-Metalium output against the CPU/PyTorch reference implementation in `tt_rqm_kernels.quaternion_ops.qmul`.

Recommended validation checks:

- elementwise numerical error against PyTorch reference output
- identity multiplication
- basis multiplication rules
- associativity within floating-point tolerance
- broadcast or flattened batch-shape handling, depending on the chosen example shape contract

### Metrics

The first benchmark should report:

- throughput
- latency
- numerical error
- scaling across input sizes

If Tenstorrent has preferred benchmark formats, this project should adopt those formats rather than inventing a separate reporting style.

The current repository includes StructuredBench as the CPU/PyTorch report generator. Future Tenstorrent backend examples should aim to emit the same `structuredbench.v1` fields when practical.

## Long-Term Direction: QuantumIR for Classical AI Compute

QuantumIR here means a classical/AI accelerator front end for selected
quantum-mechanics workloads, not a quantum-hardware proposal. The immediate ask
in this RFC remains narrow: placement guidance for a minimal `[N, 4]` structured
`qmul` kernel path.

Longer term, RQM Technologies is exploring QuantumIR as a domain-facing layer
above these kernels. It would lower selected quantum-mechanics workloads on
classical Tenstorrent/AI accelerators, including SU(2) rotations, unitary
composition, Hamiltonian evolution, phase/coherence updates, and AI
augmentation use cases, into the same structured quaternion, rotor, phase, and
tensor operators used by StructuredBench.

This does not claim that arbitrary quantum computation is efficiently
classically simulable, does not ask Tenstorrent for native quaternion hardware,
and does not replace the signal processing, physical AI, imaging, wave
simulation, and scientific computing kernel story. It is a future front end built
on the same kernel foundation.

## Questions for Tenstorrent Maintainers

1. Should this first live as a TT-Metalium programming example?
2. Should it instead be implemented as a TT-NN custom operation?
3. Is there a preferred custom-op pathway for small structured numerical kernels like this?
4. Would a TT-MLIR representation be useful later, after the TT-Metalium reference path is working?
5. Are there benchmark formats, input-size conventions, or reporting templates Tenstorrent prefers for example kernels?

## Non-Goals

This RFC is intentionally narrow.

Non-goals:

- not asking for native quaternion hardware
- not asking for a new datatype
- not proposing a new chip feature
- not defense-first
- not asking Tenstorrent to endorse RQM theory
- not claiming hardware performance without a real Tenstorrent hardware run

The proposal is about ordinary floating-point tensor kernels with structured numerical semantics.

## Next Steps

1. Open a GitHub Discussion with Tenstorrent maintainers.
2. Ask for placement guidance: TT-Metalium example, TT-NN custom op, or another preferred path.
3. Port `qmul` to TT-Metalium.
4. Validate TT-Metalium output against the CPU/PyTorch reference implementation.
5. Report throughput, latency, numerical error, and scaling.
6. Submit a draft PR or example if maintainers agree on the placement.
