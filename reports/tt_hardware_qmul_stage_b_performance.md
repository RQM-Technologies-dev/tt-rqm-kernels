# StructuredBench Report

Generated: `2026-07-14T18:52:22.120896+00:00`

Backend: `external-qmul`  Device: `tenstorrent/wormhole-device-0`  Execution: `hardware`  Stable benchmark: `false`  Dtype: `float32`  Suite: `qmul`

## Benchmark Results

| workload | items | iters | latency_ms | throughput | unit | max_abs_err | rms_err | stability | scalar_ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qmul | 4096 | 30 | 0.0558 | 73462067.32 | qmul/s | 1.049e-07 | 2.128e-08 | - | 5.586e-08 |
| qmul | 65536 | 30 | 0.0713 | 918578737.12 | qmul/s | 1.135e-07 | 2.129e-08 | - | 5.472e-08 |
| qmul | 262144 | 30 | 0.1408 | 1861503083.64 | qmul/s | 1.224e-07 | 2.136e-08 | - | 4.440e-08 |

## Conformance and Timing Integrity

| workload | implementation_class | performance_eligible | correctness_passed | validated_values | whole_output_max_abs_err | repetitions | device_median_s | device_p95_s | end_to_end_median_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qmul | multicore_tensix_sfpu_qmul | true | true | 16384 | 1.049e-07 | 10 | 1.673e-03 | 1.760e-03 | 2.932e+00 |
| qmul | multicore_tensix_sfpu_qmul | true | true | 262144 | 1.135e-07 | 10 | 2.140e-03 | 2.165e-03 | 2.891e+00 |
| qmul | multicore_tensix_sfpu_qmul | true | true | 1048576 | 1.224e-07 | 10 | 4.225e-03 | 4.268e-03 | 2.922e+00 |

## Hardware-Relevant Metrics

| workload | items | estimated_flops | estimated_flops_per_s | estimated_total_bytes | effective_gb_per_s | arithmetic_intensity |
| --- | --- | --- | --- | --- | --- | --- |
| qmul | 4096 | 3440640 | 2.057e+09 | 5898240 | 3.526 | 0.583 |
| qmul | 65536 | 55050240 | 2.572e+10 | 94371840 | 44.092 | 0.583 |
| qmul | 262144 | 220200960 | 5.212e+10 | 377487360 | 89.352 | 0.583 |

## Notes

- Methodology note: One Wormhole device 0 performance-eligible multicore/SFPU Stage B sweep; first sample, stable_benchmark=false.
- Current results use the external-qmul candidate harness. StructuredBench validates the whole output against an independent float64 golden calculation; hardware claims depend on the external command and measurement environment.
- External-qmul reports are candidate-command outputs validated by StructuredBench. They should not be read as Tenstorrent hardware performance unless the command and device are explicitly documented.
- Scalar reference checks are small deterministic spot checks used as an independent correctness contract.
- FLOP and byte counts are simple documented estimates for backend comparison, not hardware-counter measurements.
- Phase update includes transcendental-heavy sin/cos state generation; its FLOP estimate counts each transcendental call as one reported operation.
