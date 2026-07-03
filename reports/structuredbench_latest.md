# StructuredBench Report

Generated: `2026-07-03T00:50:42.596599+00:00`

Backend: `torch`  Device: `cpu`  Dtype: `float32`  Suite: `smoke`

## Benchmark Results

| workload | items | iters | latency_ms | throughput | unit | max_abs_err | rms_err | stability | scalar_ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qmul | 1024 | 5 | 0.1364 | 7504844.39 | qmul/s | 1.179e-07 | 2.886e-08 | - | 6.453e-08 |
| qrotate | 1024 | 5 | 0.3249 | 3152113.46 | rotations/s | 4.148e-07 | 7.443e-08 | 5.231e-07 | 3.300e-07 |
| qnormalize | 1024 | 5 | 0.0335 | 30533259.06 | normalizations/s | 1.014e-07 | 2.205e-08 | 9.681e-08 | 5.558e-08 |
| qinverse | 1024 | 5 | 0.0878 | 11658727.40 | inverses/s | 1.486e-06 | 6.595e-08 | 1.788e-07 | 8.558e-07 |
| phase_update | 2048 | 5 | 0.0784 | 26129313.40 | phase-updates/s | 3.465e-07 | 5.110e-08 | - | - |

## Hardware-Relevant Metrics

| workload | items | estimated_flops | estimated_flops_per_s | estimated_total_bytes | effective_gb_per_s | arithmetic_intensity |
| --- | --- | --- | --- | --- | --- | --- |
| qmul | 1024 | 143360 | 2.101e+08 | 245760 | 0.360 | 0.583 |
| qrotate | 1024 | 327680 | 2.017e+08 | 204800 | 0.126 | 1.600 |
| qnormalize | 1024 | 66560 | 3.969e+08 | 163840 | 0.977 | 0.406 |
| qinverse | 1024 | 76800 | 1.749e+08 | 163840 | 0.373 | 0.469 |
| phase_update | 2048 | 61440 | 1.568e+08 | 204800 | 0.523 | 0.300 |

## Notes

- Current results use the CPU/PyTorch reference backend.
- Committed reports are sample CPU/PyTorch reference outputs. They are included to show the report shape and outreach packet format, not to claim stable hardware performance.
- Scalar reference checks are small deterministic spot checks used as an independent correctness contract.
- FLOP and byte counts are simple documented estimates for backend comparison, not hardware-counter measurements.
- Phase update includes transcendental-heavy sin/cos state generation; its FLOP estimate counts each transcendental call as one reported operation.
