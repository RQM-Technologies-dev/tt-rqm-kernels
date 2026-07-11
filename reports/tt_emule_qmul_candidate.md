# StructuredBench Report

Generated: `2026-07-06T21:42:15.093241+00:00`

Backend: `external-qmul`  Device: `tt-emule/tt-metalium-riscv-qmul-candidate`  Execution: `emulation`  Stable benchmark: `false`  Dtype: `float32`  Suite: `qmul`

This report demonstrates the `external-qmul` candidate protocol with an emulation-labeled run. StructuredBench validates the candidate output through the current conformance contract. It is not hardware performance evidence.

This is a historical pre-integrity artifact and has not been rerun under the whole-output/metrics-v2 gate.

Final target: `reports/tt_hardware_qmul_quickstart.md`.

## Benchmark Results

| workload | items | iters | latency_ms | throughput | unit | max_abs_err | rms_err | stability | scalar_ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qmul | 32 | 1 | 2053.3500 | 15.58 | qmul/s | 1.092e-07 | 3.195e-08 | - | 7.559e-08 |
| qmul | 32 | 1 | 2085.5600 | 15.34 | qmul/s | 1.302e-07 | 3.131e-08 | - | 6.484e-08 |
| qmul | 32 | 1 | 1833.5100 | 17.45 | qmul/s | 7.044e-08 | 2.395e-08 | - | 7.044e-08 |

## Conformance and Timing Integrity

| workload | implementation_class | performance_eligible | correctness_passed | validated_values | whole_output_max_abs_err | repetitions | device_median_s | device_p95_s | end_to_end_median_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qmul | - | false | false | - | 0.000e+00 | 1 | - | - | - |
| qmul | - | false | false | - | 0.000e+00 | 1 | - | - | - |
| qmul | - | false | false | - | 0.000e+00 | 1 | - | - | - |

## Hardware-Relevant Metrics

| workload | items | estimated_flops | estimated_flops_per_s | estimated_total_bytes | effective_gb_per_s | arithmetic_intensity |
| --- | --- | --- | --- | --- | --- | --- |
| qmul | 32 | 896 | 4.364e+02 | 1536 | 0.000 | 0.583 |
| qmul | 32 | 896 | 4.296e+02 | 1536 | 0.000 | 0.583 |
| qmul | 32 | 896 | 4.887e+02 | 1536 | 0.000 | 0.583 |

## Notes

- Methodology note: Experimental TT-Metalium qmul candidate run through tt-emule Docker wrapper; reproducibility refresh while Tenstorrent Cloud hardware access is pending.
- Current results use the external-qmul candidate harness. StructuredBench validates the whole output against an independent float64 golden calculation; hardware claims depend on the external command and measurement environment.
- External-qmul reports are candidate-command outputs validated by StructuredBench. They should not be read as Tenstorrent hardware performance unless the command and device are explicitly documented.
- Scalar reference checks are small deterministic spot checks used as an independent correctness contract.
- FLOP and byte counts are simple documented estimates for backend comparison, not hardware-counter measurements.
- Phase update includes transcendental-heavy sin/cos state generation; its FLOP estimate counts each transcendental call as one reported operation.
