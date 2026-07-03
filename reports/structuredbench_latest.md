# StructuredBench Report

Generated: `2026-07-03T01:44:51.885429+00:00`

Backend: `torch`  Device: `cpu`  Dtype: `float32`  Suite: `smoke`

## Benchmark Results

| workload | items | iters | latency_ms | throughput | unit | max_abs_err | rms_err | stability | scalar_ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qmul | 1024 | 5 | 0.1380 | 7419332.71 | qmul/s | 1.179e-07 | 2.886e-08 | - | 6.453e-08 |
| qrotate | 1024 | 5 | 0.3232 | 3168148.24 | rotations/s | 4.148e-07 | 7.443e-08 | 5.231e-07 | 3.300e-07 |
| qnormalize | 1024 | 5 | 0.0366 | 28004615.88 | normalizations/s | 1.014e-07 | 2.205e-08 | 9.681e-08 | 5.558e-08 |
| qinverse | 1024 | 5 | 0.0887 | 11549688.28 | inverses/s | 1.486e-06 | 6.595e-08 | 1.788e-07 | 8.558e-07 |
| phase_update | 2048 | 5 | 0.0785 | 26085184.22 | phase-updates/s | 3.465e-07 | 5.110e-08 | - | - |

## Hardware-Relevant Metrics

| workload | items | estimated_flops | estimated_flops_per_s | estimated_total_bytes | effective_gb_per_s | arithmetic_intensity |
| --- | --- | --- | --- | --- | --- | --- |
| qmul | 1024 | 143360 | 2.077e+08 | 245760 | 0.356 | 0.583 |
| qrotate | 1024 | 327680 | 2.028e+08 | 204800 | 0.127 | 1.600 |
| qnormalize | 1024 | 66560 | 3.641e+08 | 163840 | 0.896 | 0.406 |
| qinverse | 1024 | 76800 | 1.732e+08 | 163840 | 0.370 | 0.469 |
| phase_update | 2048 | 61440 | 1.565e+08 | 204800 | 0.522 | 0.300 |

## Notes

- Current results use the CPU/PyTorch reference backend.
- Committed reports are sample CPU/PyTorch reference outputs. They are included to show the report shape and outreach packet format, not to claim stable hardware performance.
- Scalar reference checks are small deterministic spot checks used as an independent correctness contract.
- FLOP and byte counts are simple documented estimates for backend comparison, not hardware-counter measurements.
- Phase update includes transcendental-heavy sin/cos state generation; its FLOP estimate counts each transcendental call as one reported operation.
