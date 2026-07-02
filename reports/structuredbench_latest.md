# StructuredBench Report

Generated: `2026-07-02T22:39:20.265001+00:00`

Backend: `torch`  Device: `cpu`  Dtype: `float32`  Suite: `smoke`

## Benchmark Results

| workload | items | iters | latency_ms | throughput | unit | max_abs_err | rms_err | stability | scalar_ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qmul | 1024 | 5 | 0.1218 | 8410567.33 | qmul/s | 1.179e-07 | 2.886e-08 | - | 6.453e-08 |
| qrotate | 1024 | 5 | 0.3406 | 3006418.62 | rotations/s | 4.148e-07 | 7.443e-08 | 5.231e-07 | 3.300e-07 |
| qnormalize | 1024 | 5 | 0.0280 | 36580572.05 | normalizations/s | 1.014e-07 | 2.205e-08 | 9.681e-08 | 5.558e-08 |
| qinverse | 1024 | 5 | 0.0703 | 14565522.84 | inverses/s | 1.486e-06 | 6.595e-08 | 1.788e-07 | 8.558e-07 |
| phase_update | 2048 | 5 | 0.0629 | 32563242.71 | phase-updates/s | 3.465e-07 | 5.110e-08 | - | - |

## Hardware-Relevant Metrics

| workload | items | estimated_flops | estimated_flops_per_s | estimated_total_bytes | effective_gb_per_s | arithmetic_intensity |
| --- | --- | --- | --- | --- | --- | --- |
| qmul | 1024 | 143360 | 2.355e+08 | 245760 | 0.404 | 0.583 |
| qrotate | 1024 | 327680 | 1.924e+08 | 204800 | 0.120 | 1.600 |
| qnormalize | 1024 | 66560 | 4.755e+08 | 163840 | 1.171 | 0.406 |
| qinverse | 1024 | 76800 | 2.185e+08 | 163840 | 0.466 | 0.469 |
| phase_update | 2048 | 61440 | 1.954e+08 | 204800 | 0.651 | 0.300 |

## Notes

- Current results use the CPU/PyTorch reference backend.
- Scalar reference checks are small deterministic spot checks used as an independent correctness contract.
- FLOP and byte counts are simple documented estimates for backend comparison, not hardware-counter measurements.
- Phase update includes transcendental-heavy sin/cos state generation; its FLOP estimate counts each transcendental call as one reported operation.
