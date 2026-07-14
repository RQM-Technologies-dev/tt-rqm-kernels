# StructuredBench Report

Generated: `2026-07-14T18:48:14.999929+00:00`

Backend: `external-qmul`  Device: `tenstorrent/wormhole-device-0`  Execution: `hardware`  Stable benchmark: `false`  Dtype: `float32`  Suite: `qmul`

## Benchmark Results

| workload | items | iters | latency_ms | throughput | unit | max_abs_err | rms_err | stability | scalar_ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qmul | 128 | 1 | 15.6002 | 8205.02 | qmul/s | 9.687e-08 | 2.110e-08 | - | 4.585e-08 |

## Conformance and Timing Integrity

| workload | implementation_class | performance_eligible | correctness_passed | validated_values | whole_output_max_abs_err | repetitions | device_median_s | device_p95_s | end_to_end_median_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qmul | multicore_tensix_sfpu_qmul | false | true | 512 | 9.687e-08 | 1 | 1.560e-02 | 1.560e-02 | 2.879e+00 |

## Hardware-Relevant Metrics

| workload | items | estimated_flops | estimated_flops_per_s | estimated_total_bytes | effective_gb_per_s | arithmetic_intensity |
| --- | --- | --- | --- | --- | --- | --- |
| qmul | 128 | 3584 | 2.297e+05 | 6144 | 0.000 | 0.583 |

## Notes

- Methodology note: One Wormhole device 0 multicore/SFPU N=128 conformance gate; not a performance sample.
- Current results use the external-qmul candidate harness. StructuredBench validates the whole output against an independent float64 golden calculation; hardware claims depend on the external command and measurement environment.
- External-qmul reports are candidate-command outputs validated by StructuredBench. They should not be read as Tenstorrent hardware performance unless the command and device are explicitly documented.
- Scalar reference checks are small deterministic spot checks used as an independent correctness contract.
- FLOP and byte counts are simple documented estimates for backend comparison, not hardware-counter measurements.
- Phase update includes transcendental-heavy sin/cos state generation; its FLOP estimate counts each transcendental call as one reported operation.
