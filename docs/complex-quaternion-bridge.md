# ComplexTensor to QuaternionTensor Bridge

## Purpose

This document frames `tt-rqm-kernels` as a conservative extension of existing
structured tensor practice in the Tenstorrent ecosystem.

Tracker issue:
<https://github.com/RQM-Technologies-dev/tt-rqm-kernels/issues/9>

Tenstorrent TT-NN already documents complex tensor operations such as
`complex_tensor`, `real`, `imag`, `angle`, `conj`, and `polar`. RQM should use
that as the familiar analogy:

```text
ComplexTensor:    [real, imag]
QuaternionTensor: [real, i, j, k]
StructuredTensor: lane-packed physical/geometric/signal state
```

This is not a request for a native quaternion datatype. It is a software design
frame for structured values represented inside ordinary floating-point tensors.

## Recommended Framing

Lead with:

- structured tensor values, not quaternion theory
- ordinary floating-point tensors, not custom hardware types
- reproducible operators and reports, not endorsement requests
- `qmul` as the first four-lane structured operator

Avoid:

- native quaternion hardware asks
- new chip feature requests
- claims that TT-NN should adopt RQM abstractions
- speculative physics language
- hardware-performance claims from CPU, simulator, or emulation reports

## Representation Options

### Lane-Packed Tensor

Current StructuredBench contract:

```text
q: [N, 4] = [real, i, j, k]
```

Advantages:

- simple external binary protocol
- easy CPU/PyTorch validation
- compact benchmark shape
- matches current `qmul` and TT-Lang simulator reports

Tradeoff:

- a backend kernel must explicitly handle final-lane cross-dependencies

### Aligned-Lane Tensors

Alternative TT-NN-adjacent representation:

```text
real: [N]
i:    [N]
j:    [N]
k:    [N]
```

Advantages:

- resembles the way complex APIs expose real and imaginary parts
- may be easier to map into existing pointwise operations or custom-op
  internals
- makes each lane explicit for debugging and layout experiments

Tradeoff:

- less compact as a public benchmark protocol
- requires careful synchronization of four aligned tensors

## Bridge Contract

The bridge should preserve the current operator contract:

```text
qmul(a, b) -> out

a:   [N, 4] float32
b:   [N, 4] float32
out: [N, 4] float32
```

If a future TT-NN experiment uses aligned-lane tensors internally, it must still
round-trip to the lane-packed StructuredBench contract for reporting.

Minimum utility surface for a future prototype:

```python
split_quaternion_lanes(q) -> (real, i, j, k)
pack_quaternion_lanes(real, i, j, k) -> q
```

These helpers should start as CPU/PyTorch utilities or documentation examples.
Do not add TT-NN code until lower-stack `qmul` placement or maintainer guidance
is clearer.

## Validation Path

Any bridge prototype should validate:

1. `[N, 4]` lane-packed input splits into four aligned lanes.
2. Four aligned lanes pack back into byte-identical or tolerance-equivalent
   `[N, 4]` tensors.
3. `qmul` through any bridge path matches `tt_rqm_kernels.quaternion_ops.qmul`.
4. StructuredBench report fields remain unchanged.

The bridge is successful when it helps a TT-NN maintainer understand the shape
of the problem without changing the current `qmul` benchmark contract.

## Relationship To TT-NN

This bridge is a design note, not a TT-NN integration proposal.

The right TT-NN question later is:

```text
If qmul has a lower-stack backend, should a Python-facing TT-NN wrapper accept
lane-packed [N, 4] tensors, aligned-lane tensors, or both?
```

That question should wait until there is backend evidence from TT-Lang,
tt-emule, TT-Metalium, or maintainer guidance.

## Non-Goals

- No native quaternion datatype.
- No new Tenstorrent hardware feature.
- No TT-NN custom operation in this milestone.
- No claim that Tenstorrent endorses this abstraction.
- No defense-first framing.

## References

- TT-NN API index:
  <https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/api.html>
- `ttnn.angle` ComplexTensor example:
  <https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/api/ttnn.angle.html>
- `ttnn.conj` ComplexTensor example:
  <https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/api/ttnn.conj.html>
- Operator contracts: `docs/operator-contracts.md`
- StructuredBench spec: `docs/structuredbench-spec.md`
