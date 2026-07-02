# Tenstorrent Outreach Packet

## Project Summary

`tt-rqm-kernels` is an independent RQM Technologies LLC project for structured quaternion, rotor, and phase-aware tensor kernels represented inside ordinary floating-point tensors. StructuredBench provides a CPU/PyTorch reference benchmark contract intended to compare future TT-Metalium and TT-NN backend implementations.

## Benchmark Table

| workload | items | iters | latency_ms | throughput | unit | max_abs_err |
| --- | --- | --- | --- | --- | --- | --- |
| qmul | 1024 | 5 | 0.1218 | 8410567.33 | qmul/s | 1.179e-07 |
| qrotate | 1024 | 5 | 0.3406 | 3006418.62 | rotations/s | 4.148e-07 |
| qnormalize | 1024 | 5 | 0.0280 | 36580572.05 | normalizations/s | 1.014e-07 |
| qinverse | 1024 | 5 | 0.0703 | 14565522.84 | inverses/s | 1.486e-06 |
| phase_update | 2048 | 5 | 0.0629 | 32563242.71 | phase-updates/s | 3.465e-07 |

## Hardware Metrics Table

| workload | items | estimated_flops | estimated_flops_per_s | estimated_total_bytes | effective_gb_per_s | arithmetic_intensity |
| --- | --- | --- | --- | --- | --- | --- |
| qmul | 1024 | 143360 | 2.355e+08 | 245760 | 0.404 | 0.583 |
| qrotate | 1024 | 327680 | 1.924e+08 | 204800 | 0.120 | 1.600 |
| qnormalize | 1024 | 66560 | 4.755e+08 | 163840 | 1.171 | 0.406 |
| qinverse | 1024 | 76800 | 2.185e+08 | 163840 | 0.466 | 0.469 |
| phase_update | 2048 | 61440 | 1.954e+08 | 204800 | 0.651 | 0.300 |

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

RQM Technologies has a CPU/PyTorch reference benchmark for structured quaternion and rotor tensor kernels, with qmul as the proposed first [N, 4] TT-Metalium target and qrotate_vector as the proposed second target.

We have a CPU/PyTorch reference benchmark and want guidance on the right first Tenstorrent integration point. Should this begin as a TT-Metalium programming example, a TT-NN custom operator, or another path?

The benchmark reports throughput, latency, numerical error, estimated FLOPs/sec, effective GB/sec, and arithmetic intensity, with scalar-reference spot checks for correctness.
```

## Suggested Discord Post

```text
Hi Tenstorrent community, RQM Technologies is building an independent structured-kernel benchmark for quaternion and rotor tensor operators represented inside ordinary floating-point tensors.

For a first external structured-kernel contribution, should we target a TT-Metalium programming example or a TT-NN custom-op path?

Repo: https://github.com/RQM-Technologies-dev/tt-rqm-kernels
```
