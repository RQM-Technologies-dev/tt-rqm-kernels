# StructuredBench

StructuredBench is the benchmark suite for `tt-rqm-kernels`.

It turns the reference quaternion, rotor, inverse, normalization, and phase utilities into a consistent benchmark report that can later compare CPU/PyTorch, TT-Metalium, TT-NN, and other backend implementations.

The benchmark target is lower-stack numerical infrastructure:

- structured 4-lane tensor values
- quaternion multiplication
- streamed rotor/vector rotation
- normalization and inverse stability
- phase and orientation update patterns
- reports suitable for backend comparison

It is not defense-first and does not make speculative physics claims. The workload class is relevant to robotics, graphics, wireless, imaging, wave simulation, physical AI, scientific computing, signal processing, and defense as one downstream application area.

## Commands

Run a small end-to-end smoke suite:

```bash
python -m tt_rqm_kernels.structuredbench --suite smoke
```

Run the full CPU/PyTorch reference suite:

```bash
python -m tt_rqm_kernels.structuredbench --suite full
```

Run focused suites:

```bash
python -m tt_rqm_kernels.structuredbench --suite qmul
python -m tt_rqm_kernels.structuredbench --suite qrotate
```

Emit JSON for comparison pipelines:

```bash
python -m tt_rqm_kernels.structuredbench --suite smoke --format json
```

Write a report file:

```bash
python -m tt_rqm_kernels.structuredbench --suite full --format json --output structuredbench.json
```

Use smaller inputs for local development:

```bash
python -m tt_rqm_kernels.structuredbench --suite full --items 1024 --iters 3 --warmup 1
```

## Suites

`smoke` runs one small case for each core workload:

- `qmul`
- `qrotate`
- `qnormalize`
- `qinverse`
- `phase_update`

`full` runs scaled cases across those workloads.

`qmul` runs only Hamilton product cases over tensors shaped `[N, 4]`.

`qrotate` runs only streamed rotor/vector rotation cases over rotors shaped `[N, 4]` and vectors shaped `[N, 3]`.

## Report Schema

StructuredBench emits a versioned report:

```text
schema = structuredbench.v1
```

Each result includes:

- suite
- workload
- backend
- device
- dtype
- item count
- iterations and warmup
- structured shape
- elapsed seconds
- latency per iteration
- throughput
- maximum absolute error
- maximum relative error
- RMS error
- optional stability metric
- checksum

The current backend is `torch`. Future TT-Metalium and TT-NN benchmark paths should emit the same fields so reports can be compared directly.

## Workload Notes

`qmul` measures Hamilton product throughput for structured 4-lane quaternion tensors.

`qrotate` measures streamed vector rotation by unit rotors and reports norm-preservation error as its stability metric.

`qnormalize` measures normalization of scaled quaternion tensors and reports unit-norm error.

`qinverse` measures quaternion inverse throughput and reports the residual error of `q * inverse(q)` against the identity quaternion.

`phase_update` measures wrapped phase integration and conversion to a `[cos, sin]` state vector.

## Tenstorrent Relevance

StructuredBench is intended to make future Tenstorrent backend work concrete:

1. Keep CPU/PyTorch outputs as the correctness reference.
2. Port selected workloads to TT-Metalium or TT-NN.
3. Emit the same `structuredbench.v1` report fields.
4. Compare throughput, latency, numerical error, and scaling across backends.
