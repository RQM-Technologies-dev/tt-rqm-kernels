# Structured Tensor Algebra

Structured tensor algebra treats an ordinary dense tensor as a carrier for values with additional algebraic meaning. The tensor runtime still sees floating-point arrays, shapes, strides, broadcasts, and tiled memory movement. The library layer supplies operators that interpret one or more axes as structured state.

For `tt-rqm-kernels`, the first structured states are quaternions, rotors, phase values, orientation deltas, and simple wave-state quantities.

## Scalar, Complex, and Quaternion Tensors

A scalar tensor stores one real floating-point value per logical element:

```text
shape = [...]
value = x
```

A complex tensor can be represented as two real floating-point values per logical element:

```text
shape = [..., 2]
value = [real, imag]
```

A quaternion tensor uses four real floating-point values per logical element:

```text
shape = [..., 4]
value = [real, i, j, k]
```

The final dimension stores the structured value. The leading dimensions are batch, spatial, sequence, tile, feature, or stream dimensions. This means quaternion operators can use normal tensor broadcasting over the leading dimensions while enforcing final dimension size `4`.

## Quaternions Living Inside Floats

In this project, a quaternion is not a custom Python object and not a separate runtime primitive. It is a real floating-point tensor whose final dimension is interpreted as:

```text
[w, x, y, z]
```

The Hamilton product is then implemented as a structured tensor operation over that final dimension. Rotors use unit quaternions to represent rotations, and vector rotation is implemented with:

```text
v' = r * [0, v] * conjugate(r)
```

The result is a tensor-native reference path that can be validated on CPU before lower-stack accelerator kernels are written.

## Why This Fits Accelerator Work

Accelerator runtimes are good at moving and multiplying tensors. Domain software often needs more than scalar multiply-add: rotations, phase wrapping, orientation deltas, signal phase updates, and wave-state mixing all have structure.

Structured tensor kernels give those operations explicit names, validation rules, and benchmarks while keeping the storage representation compatible with dense tensor hardware.

## Application Areas

This library is intended as lower-stack numerical infrastructure that can be useful across multiple domains:

- robotics: pose updates, orientation streams, sensor fusion reference math
- graphics: rotation streams, transform interpolation, quaternion animation state
- wireless: phase tracking, carrier phase deltas, signal orientation features
- imaging: structured transforms, phase-aware image and sensor pipelines
- wave simulation: compact phase and orientation state updates
- AI and physical AI: feature representations that carry orientation or phase
- scientific computing: reference kernels for structured numerical experiments
- signal processing: phase wrapping, phase deltas, and quaternion-valued features
- defense: downstream use where these numerical patterns are relevant

The project does not rely on speculative physics claims. Its first scope is conventional numerical software: define tensor conventions, implement reference kernels, test them, and make later hardware ports measurable.
