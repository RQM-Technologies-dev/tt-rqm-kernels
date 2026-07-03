# StructuredBench Report

Generated: `2026-07-03T17:46:43.965353+00:00`

Backend: `tt-lang-sim`  Device: `functional-simulator`  Dtype: `float32`  Suite: `qmul`

## Benchmark Results

| workload | items | iters | latency_ms | throughput | unit | max_abs_err | rms_err | stability | scalar_ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qmul | 128 | 1 | 18.4083 | 6953.38 | qmul/s | 1.055e-07 | 2.865e-08 | - | 6.278e-08 |

## Hardware-Relevant Metrics

| workload | items | estimated_flops | estimated_flops_per_s | estimated_total_bytes | effective_gb_per_s | arithmetic_intensity |
| --- | --- | --- | --- | --- | --- | --- |
| qmul | 128 | 3584 | 1.947e+05 | 6144 | 0.000 | 0.583 |

## Notes

- Current results use the TT-Lang functional simulator. They validate kernel logic and report shape, not hardware performance.
- This committed TT-Lang report is a simulator smoke output. It is included to show the report shape, not to claim stable hardware performance.
- Scalar reference checks are small deterministic spot checks used as an independent correctness contract.
- FLOP and byte counts are simple documented estimates for backend comparison, not hardware-counter measurements.
- Phase update includes transcendental-heavy sin/cos state generation; its FLOP estimate counts each transcendental call as one reported operation.
