# Wormhole qmul output-backpressure ablation

Only output circular-buffer depth changed. Both runs used the same binary, arithmetic, layout, 56-core allocation, protocol, and timing contract.

| N | depth 2 median ms | depth 4 median ms | D4/D2 | depth 2 p95 ms | depth 4 p95 ms | D4/D2 p95 | correct |
|---:|---:|---:|---:|---:|---:|---:|---|
| 65,536 | 2.065414 | 2.064508 | 0.999561 | 2.090238 | 2.094109 | 1.001852 | yes |
| 262,144 | 4.182311 | 4.211616 | 1.007007 | 4.250616 | 4.258526 | 1.001861 | yes |

Decision: retain depth 2. Depth 4 was effectively unchanged at N=65,536 and slower at N=262,144.

A pre-device setup failure caused by an unset TT_METAL_RUNTIME_ROOT is preserved in raw evidence and excluded from timing comparison.
