# Tenstorrent Outreach Packet

## Project Summary

`tt-rqm-kernels` is an independent RQM Technologies LLC project for structured quaternion, rotor, and phase-aware tensor kernels represented inside ordinary floating-point tensors. StructuredBench provides a CPU/PyTorch reference benchmark contract intended to compare future TT-Metalium and TT-NN backend implementations.

Committed reports are sample CPU/PyTorch reference outputs. They are included to show the report shape and outreach packet format, not to claim stable hardware performance.

Report labels:

```text
execution_label: cpu
stable_benchmark: false
methodology_note: CPU/PyTorch reference run; not a hardware performance result.
```

## Why Tenstorrent Developers Should Care

StructuredBench gives Tenstorrent a compact benchmark class between scalar elementwise ops and large matmul. It focuses on structured 4-lane tensor values that carry rotation, phase, orientation, direction, and geometric state inside ordinary floating-point tensors.

The first target is `qmul` over `[N, 4]` tensors. It is small enough to validate with CPU/PyTorch and scalar references, but structured enough to exercise cross-lane dependencies, fixed multiply/add/sign patterns, data movement, fusion, register reuse, and arithmetic intensity. No native quaternion datatype, new silicon feature, or hardware change is required.

Proof path:

```text
CPU/PyTorch qmul reference
-> scalar correctness check
-> TT-Metalium qmul for [N, 4]
-> compare throughput, latency, numerical error, FLOPs/sec, GB/sec, and arithmetic intensity
```

## Benchmark Table

| workload | items | iters | latency_ms | throughput | unit | max_abs_err |
| --- | --- | --- | --- | --- | --- | --- |
| qmul | 1024 | 5 | 0.1719 | 5955884.68 | qmul/s | 1.179e-07 |
| qrotate | 1024 | 5 | 0.4109 | 2491841.90 | rotations/s | 4.148e-07 |
| qnormalize | 1024 | 5 | 0.0432 | 23694159.54 | normalizations/s | 1.014e-07 |
| qinverse | 1024 | 5 | 0.2534 | 4040758.04 | inverses/s | 1.486e-06 |
| phase_update | 2048 | 5 | 0.1506 | 13595218.36 | phase-updates/s | 3.465e-07 |

## Hardware Metrics Table

| workload | items | estimated_flops | estimated_flops_per_s | estimated_total_bytes | effective_gb_per_s | arithmetic_intensity |
| --- | --- | --- | --- | --- | --- | --- |
| qmul | 1024 | 143360 | 1.668e+08 | 245760 | 0.286 | 0.583 |
| qrotate | 1024 | 327680 | 1.595e+08 | 204800 | 0.100 | 1.600 |
| qnormalize | 1024 | 66560 | 3.080e+08 | 163840 | 0.758 | 0.406 |
| qinverse | 1024 | 76800 | 6.061e+07 | 163840 | 0.129 | 0.469 |
| phase_update | 2048 | 61440 | 8.157e+07 | 204800 | 0.272 | 0.300 |

## Proposed First TT-Metalium Target

Proposed first TT-Metalium target: `qmul` for `[N, 4]` quaternion tensors.

## Proposed Second Target

Proposed second target: `qrotate_vector` for streamed unit-rotor/vector rotation.

## Relevant Docs

- [docs/operator-contracts.md](../docs/operator-contracts.md)
- [docs/structuredbench-spec.md](../docs/structuredbench-spec.md)
- [docs/tenstorrent-rfc.md](../docs/tenstorrent-rfc.md)

## Suggested GitHub Discussion Text

```text
Hi Tenstorrent maintainers,

RQM Technologies has a CPU/PyTorch reference benchmark for structured quaternion and rotor tensor kernels, with qmul as the proposed first [N, 4] TT-Metalium target.

Where should a minimal TT-Metalium qmul example for [N, 4] structured tensors live?

Secondary questions: if a TT-Metalium programming example is not the right starting point, is there a preferred TT-NN custom-op path? Would a TT-MLIR representation be useful later, after there is a concrete lower-stack qmul example?

The benchmark reports throughput, latency, numerical error, estimated FLOPs/sec, effective GB/sec, and arithmetic intensity, with scalar-reference spot checks for correctness.
```

## Suggested Discord Post

```text
Hi Tenstorrent community, RQM Technologies is building an independent structured-kernel benchmark for quaternion and rotor tensor operators represented inside ordinary floating-point tensors.

For a first external structured-kernel contribution, should we target a TT-Metalium programming example or a TT-NN custom-op path?

Repo: https://github.com/RQM-Technologies-dev/tt-rqm-kernels
```
