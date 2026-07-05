# StructuredBench Report

Generated: `2026-07-04T13:04:06.189296+00:00`

Backend: `external-qmul`  Device: `tt-emule/tt-metalium-riscv-qmul-candidate`  Execution: `emulation`  Stable benchmark: `false`  Dtype: `float32`  Suite: `qmul`

This report demonstrates the `external-qmul` candidate protocol with an emulation-labeled run. StructuredBench validates the candidate output against CPU/PyTorch plus scalar references. It is not hardware performance evidence.

Final target: `reports/tt_hardware_qmul_quickstart.md`.

## Benchmark Results

| workload | items | iters | latency_ms | throughput | unit | max_abs_err | rms_err | stability | scalar_ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qmul | 32 | 1 | 2039.2000 | 15.69 | qmul/s | 1.092e-07 | 3.195e-08 | - | 7.559e-08 |
| qmul | 32 | 1 | 2222.6200 | 14.40 | qmul/s | 1.302e-07 | 3.131e-08 | - | 6.484e-08 |
| qmul | 32 | 1 | 2485.9800 | 12.87 | qmul/s | 7.044e-08 | 2.395e-08 | - | 7.044e-08 |

## Hardware-Relevant Metrics

| workload | items | estimated_flops | estimated_flops_per_s | estimated_total_bytes | effective_gb_per_s | arithmetic_intensity |
| --- | --- | --- | --- | --- | --- | --- |
| qmul | 32 | 896 | 4.394e+02 | 1536 | 0.000 | 0.583 |
| qmul | 32 | 896 | 4.031e+02 | 1536 | 0.000 | 0.583 |
| qmul | 32 | 896 | 3.604e+02 | 1536 | 0.000 | 0.583 |

## Notes

- Methodology note: Experimental TT-Metalium qmul candidate run through tt-emule Docker wrapper; first validation sample, not a stable hardware benchmark.
- Current results use the external-qmul candidate harness. StructuredBench validates candidate output against CPU/PyTorch and scalar references; hardware claims depend on the external command and measurement environment.
- External-qmul reports are candidate-command outputs validated by StructuredBench. They should not be read as Tenstorrent hardware performance unless the command and device are explicitly documented.
- Scalar reference checks are small deterministic spot checks used as an independent correctness contract.
- FLOP and byte counts are simple documented estimates for backend comparison, not hardware-counter measurements.
- Phase update includes transcendental-heavy sin/cos state generation; its FLOP estimate counts each transcendental call as one reported operation.
