# StructuredBench Report

Generated: `2026-07-03T19:43:44.015086+00:00`

Backend: `tt-lang-sim`  Device: `functional-simulator`  Execution: `simulator`  Stable benchmark: `false`  Dtype: `float32`  Suite: `qmul`

This report demonstrates that the `[N, 4]` `qmul` contract can be exercised through the TT-Lang functional simulator and validated against CPU/PyTorch plus scalar references. It is a logic and report-shape artifact, not hardware performance evidence.

Next evidence target: `reports/tt_emule_qmul_candidate.md`.
Final target: `reports/tt_hardware_qmul_quickstart.md`.

## Benchmark Results

| workload | items | iters | latency_ms | throughput | unit | max_abs_err | rms_err | stability | scalar_ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qmul | 128 | 1 | 19.0821 | 6707.84 | qmul/s | 1.055e-07 | 2.865e-08 | - | 6.278e-08 |

## Hardware-Relevant Metrics

| workload | items | estimated_flops | estimated_flops_per_s | estimated_total_bytes | effective_gb_per_s | arithmetic_intensity |
| --- | --- | --- | --- | --- | --- | --- |
| qmul | 128 | 3584 | 1.878e+05 | 6144 | 0.000 | 0.583 |

## Notes

- Methodology note: TT-Lang functional simulator run; not hardware performance.
- Current results use the TT-Lang functional simulator. They validate kernel logic and report shape, not hardware performance.
- This committed TT-Lang report is a simulator smoke output. It is included to show the report shape, not to claim stable hardware performance.
- Simulator metadata: seed=0, layout=row-major, block_items=32, padded_items=128, variant=block-slice, sim_cli=tt-lang-sim, sim_version=tt-lang-sim 1.1.3, stats_cli=None, trace_enabled=False.
- Scalar reference checks are small deterministic spot checks used as an independent correctness contract.
- FLOP and byte counts are simple documented estimates for backend comparison, not hardware-counter measurements.
- Phase update includes transcendental-heavy sin/cos state generation; its FLOP estimate counts each transcendental call as one reported operation.
