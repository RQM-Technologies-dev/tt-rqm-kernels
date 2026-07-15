# RFC: Upstream Placement For FP32 qmul

## Summary

RQM Technologies LLC maintains a hardware-qualified FP32 Hamilton-product
kernel for Tenstorrent Wormhole. This RFC asks one placement question: should a
minimal upstream version live as a TT-Metalium programming example or as an
experimental TT-NN device operation using `ProgramDescriptor`?

The request is open as
[tenstorrent/tt-metal#49887](https://github.com/tenstorrent/tt-metal/issues/49887).

This is an independent community proposal, not an endorsement claim or a
request for a native quaternion datatype or hardware feature.

## Operator

```text
qmul(a, b) -> output

a:      FP32 [N, 4]
b:      FP32 [N, 4]
output: FP32 [N, 4]
lanes:  [real, i, j, k]
```

For `a = [aw, ax, ay, az]` and `b = [bw, bx, by, bz]`:

```text
output.w = aw*bw - ax*bx - ay*by - az*bz
output.x = aw*bx + ax*bw + ay*bz - az*by
output.y = aw*by - ax*bz + ay*bw + az*bx
output.z = aw*bz + ax*by - ay*bx + az*bw
```

Multiplication order is observable: `qmul(a, b)` is generally not equal to
`qmul(b, a)`.

## Existing Evidence

The current external TT-Metalium implementation has:

- whole-output validation against an independent Float64 Hamilton-product
  golden with `atol=1e-4` and `rtol=1e-4`;
- basis, identity, and noncommutative-order tests;
- multicore Tensix compute/SFPU execution on Wormhole;
- three qualified N300 device-0 sessions supporting an aggregate Claim Level 2
  release; and
- Device Program Profiler and Tracy diagnostics.

The evidence does not establish CPU acceleration, measured DRAM/NoC/PCIe
bandwidth, application speedup, energy efficiency, or dual-device scaling.

Primary references:

- [operator contract](operator-contracts.md#qmul)
- [Claim Level 2 report](benchmarks/wormhole-qmul.md)
- [profiler and hardware diagnostics](benchmarks/wormhole-qmul-hardware-evidence.md)
- [standalone multicore host source](../experimental/tt_metalium_qmul/src/qmul_multicore_candidate.cpp)
- [reader, compute, and writer kernels](../experimental/tt_metalium_qmul/kernels/)

## Placement Decision Requested

> We have a hardware-qualified FP32 `[N,4]` Hamilton-product kernel on
> Wormhole with whole-output validation, three-session stability evidence, and
> profiler diagnostics. Would Tenstorrent prefer a minimal upstream version as
> a TT-Metalium programming example or as an experimental TT-NN device
> operation using ProgramDescriptor?

The qualified standalone candidate directly constructs TT-Metalium programs,
circular buffers, kernels, runtime arguments, buffers, and a `MeshWorkload`.
If TT-NN is preferred, the port would instead follow the current device-
operation structure, expose a Python binding, and attach the existing PyTorch
implementation as its golden function.

## Layout Question

The public contract is row-major `[N, 4]`, but the qualified device path packs
the four components into separate sequences of padded 32x32 FP32 tiles. For an
experimental TT-NN operation, should the public API:

1. accept logical `[N, 4]` and perform an internal layout adaptation; or
2. accept the component-planar tiled representation directly?

The first upstream implementation should preserve the answer chosen by
maintainers rather than silently changing the qualified kernel's data path.

## Proposed Validation

Whichever placement is selected, the port will retain:

- the PyTorch Hamilton product as the golden reference;
- whole-output FP32 validation and nonfinite rejection;
- basis multiplication and reversed-order tests;
- representative Wormhole sizes `N=4096`, `65536`, and `262144`; and
- one diagnostic profiler capture for the upstream-shaped implementation.

The new current-`main` port will not inherit `stable_benchmark=true` from the
existing pinned release. No upstream PR will be opened until the feature issue
establishes placement and expected layout.

## Non-Goals

- no native quaternion datatype;
- no silicon or instruction-set change;
- no CPU comparison or acceleration claim;
- no SU2, Hamiltonian-simulation, compiler-IR, entanglement, or broader
  application proposal; and
- no Tenstorrent endorsement claim.
