# StructuredBench Report

Generated: `2026-07-14T18:14:01.300014+00:00`

Backend: `external-qmul`  Device: `tt-metalium-riscv-qmul-candidate`  Execution: `hardware`  Stable benchmark: `false`  Dtype: `float32`  Suite: `qmul`

## Benchmark Results

| workload | items | iters | latency_ms | throughput | unit | max_abs_err | rms_err | stability | scalar_ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qmul | 128 | 1 | 4.8934 | 26157.90 | qmul/s | 7.709e-08 | 2.243e-08 | - | 7.709e-08 |

## Conformance and Timing Integrity

| workload | implementation_class | performance_eligible | correctness_passed | validated_values | whole_output_max_abs_err | repetitions | device_median_s | device_p95_s | end_to_end_median_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qmul | scalar_riscv_correctness_baseline | false | true | 512 | 7.709e-08 | 1 | 4.893e-03 | 4.893e-03 | 2.900e+00 |

## Hardware-Relevant Metrics

| workload | items | estimated_flops | estimated_flops_per_s | estimated_total_bytes | effective_gb_per_s | arithmetic_intensity |
| --- | --- | --- | --- | --- | --- | --- |
| qmul | 128 | 3584 | 7.324e+05 | 6144 | 0.001 | 0.583 |

## Notes

- Methodology note: Configured Tenstorrent hardware external-qmul run; first samples should not be treated as stable benchmark results unless separately validated.
- Current results use the external-qmul candidate harness. StructuredBench validates the whole output against an independent float64 golden calculation; hardware claims depend on the external command and measurement environment.
- External-qmul reports are candidate-command outputs validated by StructuredBench. They should not be read as Tenstorrent hardware performance unless the command and device are explicitly documented.
- Scalar reference checks are small deterministic spot checks used as an independent correctness contract.
- FLOP and byte counts are simple documented estimates for backend comparison, not hardware-counter measurements.
- Phase update includes transcendental-heavy sin/cos state generation; its FLOP estimate counts each transcendental call as one reported operation.
