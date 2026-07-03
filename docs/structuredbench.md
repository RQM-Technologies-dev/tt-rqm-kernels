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

Write JSON and Markdown report files:

```bash
python -m tt_rqm_kernels.structuredbench \
  --suite smoke \
  --json-output reports/structuredbench_latest.json \
  --markdown-output reports/structuredbench_latest.md
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
- scalar reference spot-check error
- estimated FLOPs
- estimated FLOPs/sec
- estimated bytes read
- estimated bytes written
- estimated total bytes
- effective GB/sec
- arithmetic intensity in FLOPs/byte
- checksum

The current backend is `torch`. Future TT-Metalium and TT-NN benchmark paths should emit the same fields so reports can be compared directly.

The optional TT-Lang simulator qmul smoke also emits `structuredbench.v1` with
`backend="tt-lang-sim"` and `simulation=true`. Those reports validate simulator
logic and report shape; they are not hardware performance results.

Run the optional simulator backend through the StructuredBench CLI when
`tt-lang-sim` is installed:

```bash
python -m tt_rqm_kernels.structuredbench \
  --backend tt-lang-sim \
  --suite qmul \
  --items 128 \
  --iters 1 \
  --warmup 0
```

## Hardware Metric Estimates

StructuredBench reports simple hardware-relevant estimates. These are not hardware-counter measurements; they are contract-level estimates for comparing CPU/PyTorch and future accelerator backend reports.

`qmul`:

- 28 FLOPs per Hamilton product
- reads two 4-lane quaternion inputs
- writes one 4-lane quaternion output
- bytes per value comes from dtype

`qrotate`:

- 64 estimated FLOPs per rotated vector
- counts two Hamilton products plus conservative conjugate/vector-packing overhead
- logical traffic reads one rotor `[N, 4]` and one vector `[N, 3]`, then writes one vector `[N, 3]`

`qnormalize`:

- 13 estimated FLOPs per quaternion
- counts norm, reciprocal/division or scaling, and output write traffic

`qinverse`:

- 15 estimated FLOPs per quaternion
- counts conjugate, norm squared, reciprocal/division or scaling, and output write traffic

`phase_update`:

- 6 estimated FLOPs per item
- phase integration plus sin/cos state generation and amplitude scaling
- transcendental-heavy behavior is marked in reports and should be interpreted separately from simple fused multiply-add throughput

## Workload Notes

`qmul` measures Hamilton product throughput for structured 4-lane quaternion tensors.

`qrotate` measures streamed vector rotation by unit rotors and reports norm-preservation error as its stability metric.

`qnormalize` measures normalization of scaled quaternion tensors and reports unit-norm error.

`qinverse` measures quaternion inverse throughput and reports the residual error of `q * inverse(q)` against the identity quaternion.

`phase_update` measures wrapped phase integration and conversion to a `[cos, sin]` state vector.

## Tenstorrent Relevance

StructuredBench is intended to make future Tenstorrent backend work concrete:

1. Keep CPU/PyTorch outputs as the correctness reference.
2. Validate `qmul` in TT-Lang simulation.
3. Port selected workloads to TT-Metalium or TT-NN.
4. Emit the same `structuredbench.v1` report fields.
5. Compare throughput, latency, numerical error, and scaling across backends.
