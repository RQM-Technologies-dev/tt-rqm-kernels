# Operator Contracts

This document defines the reference contracts for the structured tensor operators in `tt-rqm-kernels`.

The common quaternion convention is:

```text
[..., 4] = [real, i, j, k]
```

All current implementations are CPU/PyTorch reference kernels. Future TT-Metalium, TT-NN, or TT-MLIR paths should match these contracts before adding backend-specific optimizations.

## Common Expectations

- Inputs are floating-point tensors.
- Quaternion tensors use final dimension size `4`.
- Vector tensors use final dimension size `3`.
- Leading dimensions follow normal PyTorch broadcasting rules unless otherwise stated.
- Current benchmark tolerances are engineering tolerances for float32 and float64 comparison, not formal numerical-analysis guarantees.
- The scalar reference backend provides small independent spot checks for selected operators.

## `qmul`

Input shape:

```text
a: [..., 4]
b: [..., 4]
```

Output shape:

```text
out: broadcast(a.shape[:-1], b.shape[:-1]) + [4]
```

Dtype expectations:

- floating-point tensor inputs
- current StructuredBench dtypes: `float32`, `float64`

Broadcasting behavior:

- leading dimensions broadcast with PyTorch rules
- final dimension must be exactly `4`

Reference equation:

```text
out.w = a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z
out.x = a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y
out.y = a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x
out.z = a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w
```

Numerical tolerance expectation:

- float32 backend results should match the CPU/PyTorch and scalar references to small absolute error on representative inputs
- float64 backend results should match more tightly

Benchmark relevance:

- first target for TT-Metalium because it is compact, structured, and uses 4-lane tensor values
- estimated 28 FLOPs per Hamilton product

Future Tenstorrent backend notes:

- first layout should be `[N, 4]`
- compare throughput, latency, numerical error, scaling, effective GB/s, and arithmetic intensity against StructuredBench CPU/PyTorch reports

## `qconj`

Input shape:

```text
q: [..., 4]
```

Output shape:

```text
out: [..., 4]
```

Dtype expectations:

- floating-point tensor input

Broadcasting behavior:

- no binary broadcasting; shape is preserved

Reference equation:

```text
qconj([w, x, y, z]) = [w, -x, -y, -z]
```

Numerical tolerance expectation:

- exact sign change aside from dtype representation

Benchmark relevance:

- used inside inverse and rotor rotation paths

Future Tenstorrent backend notes:

- likely useful as a fused subexpression rather than a standalone first backend target

## `qnorm`

Input shape:

```text
q: [..., 4]
```

Output shape:

```text
out: [...]
```

Dtype expectations:

- floating-point tensor input

Broadcasting behavior:

- shape is reduced over the quaternion lane dimension

Reference equation:

```text
qnorm(q) = sqrt(w*w + x*x + y*y + z*z)
```

Numerical tolerance expectation:

- tolerance depends on dtype and reduction behavior

Benchmark relevance:

- basis for normalization, inverse stability, and unit-rotor validation

Future Tenstorrent backend notes:

- useful as a fused component of `qnormalize`, `qinverse`, and rotor validation

## `qnormalize`

Input shape:

```text
q: [..., 4]
```

Output shape:

```text
out: [..., 4]
```

Dtype expectations:

- floating-point tensor input
- zero or near-zero quaternions are invalid

Broadcasting behavior:

- shape is preserved

Reference equation:

```text
qnormalize(q) = q / qnorm(q)
```

Numerical tolerance expectation:

- output norm should be close to `1`
- StructuredBench reports max unit-norm error as a stability metric

Benchmark relevance:

- stresses reduction, reciprocal/division, and output scaling for 4-lane values
- estimate: 13 FLOPs per item

Future Tenstorrent backend notes:

- useful after `qmul` because it validates reduction and scaling behavior

## `qinverse`

Input shape:

```text
q: [..., 4]
```

Output shape:

```text
out: [..., 4]
```

Dtype expectations:

- floating-point tensor input
- zero or near-zero quaternions are invalid

Broadcasting behavior:

- shape is preserved

Reference equation:

```text
qinverse(q) = qconj(q) / dot(q, q)
```

Numerical tolerance expectation:

- `qmul(q, qinverse(q))` should be close to `[1, 0, 0, 0]`
- StructuredBench reports this residual as a stability metric

Benchmark relevance:

- combines dot product, conjugation, reciprocal/division, and 4-lane output
- estimate: 15 FLOPs per item

Future Tenstorrent backend notes:

- good stability benchmark after standalone `qmul`

## `qdot`

Input shape:

```text
a: [..., 4]
b: [..., 4]
```

Output shape:

```text
out: broadcast(a.shape[:-1], b.shape[:-1])
```

Dtype expectations:

- floating-point tensor inputs

Broadcasting behavior:

- leading dimensions broadcast with PyTorch rules
- final dimension must be exactly `4`

Reference equation:

```text
qdot(a, b) = a.w*b.w + a.x*b.x + a.y*b.y + a.z*b.z
```

Numerical tolerance expectation:

- reduction tolerance depends on dtype

Benchmark relevance:

- basis for inverse and normalization checks

Future Tenstorrent backend notes:

- likely useful as a fused component rather than the first standalone target

## `qrotate_vector`

Input shape:

```text
rotor: [..., 4]
vector: [..., 3]
```

Output shape:

```text
out: broadcast(rotor.shape[:-1], vector.shape[:-1]) + [3]
```

Dtype expectations:

- floating-point tensor inputs
- rotor should be unit length unless validation is intentionally bypassed

Broadcasting behavior:

- leading dimensions broadcast through the underlying quaternion products

Reference equation:

```text
qrotate_vector(r, v) = (r * [0, v] * qconj(r)).xyz
```

Numerical tolerance expectation:

- vector norm should be preserved by unit rotors within dtype tolerance
- StructuredBench reports norm-preservation error as a stability metric

Benchmark relevance:

- streamed rotor/vector rotation models a common geometric workload
- estimate: two Hamilton products plus conservative conjugate/vector-packing overhead

Future Tenstorrent backend notes:

- proposed second target after `qmul`
- useful for testing fused structured kernels beyond one Hamilton product

## Phase Update Helpers

Representative helper:

```text
phase_update(phase, angular_rate, amplitude, dt)
```

Input shape:

```text
phase: [N]
angular_rate: [N]
amplitude: [N]
```

Output shape:

```text
state: [N, 2]
```

Dtype expectations:

- floating-point tensor inputs

Broadcasting behavior:

- current benchmark uses matching `[N]` inputs
- helper functions use normal PyTorch elementwise broadcasting where applicable

Reference equation:

```text
next_phase = wrap_phase(phase + angular_rate * dt)
state = amplitude * [cos(next_phase), sin(next_phase)]
```

Numerical tolerance expectation:

- tolerance depends on dtype and transcendental function implementation

Benchmark relevance:

- models phase/orientation update patterns common in signal and wave-state pipelines
- transcendental-heavy behavior is reported explicitly in StructuredBench notes

Future Tenstorrent backend notes:

- not the first proposed TT-Metalium target
- useful later for evaluating phase-aware tensor workloads and transcendental-heavy paths
