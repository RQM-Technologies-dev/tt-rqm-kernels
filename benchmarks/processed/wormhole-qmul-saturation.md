# Wormhole qmul larger-size saturation sweep

Diagnostic evidence only. Logical GB/s and estimated FLOP/s use the workload model; they are not measured fabric bandwidth or a hardware peak claim.

Memory preflight passed; the largest case uses 48 MiB across the two inputs and one output planar buffer.

| N | tiles | cores | median ms | p95 ms | qmul/s | logical GB/s | estimated GFLOP/s | max abs error |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 1 | 1 | 1.592354 | 1.610265 | 19,292,193 | 0.926 | 0.540 | 1.107e-06 |
| 4,096 | 4 | 4 | 1.602264 | 1.686744 | 76,691,482 | 3.681 | 2.147 | 7.663e-07 |
| 16,384 | 16 | 16 | 1.615534 | 1.734583 | 304,246,150 | 14.604 | 8.519 | 9.229e-07 |
| 57,344 | 56 | 56 | 1.743912 | 1.813833 | 986,471,512 | 47.351 | 27.621 | 1.266e-06 |
| 65,536 | 64 | 56 | 2.066075 | 2.145429 | 951,601,697 | 45.677 | 26.645 | 1.542e-06 |
| 131,072 | 128 | 56 | 2.511186 | 2.548405 | 1,565,857,726 | 75.161 | 43.844 | 1.275e-06 |
| 262,144 | 256 | 56 | 4.138704 | 4.300938 | 1,900,188,815 | 91.209 | 53.205 | 1.487e-06 |
| 524,288 | 512 | 56 | 5.661990 | 5.745803 | 2,777,935,212 | 133.341 | 77.782 | 1.474e-06 |
| 1,048,576 | 1024 | 56 | 10.512042 | 10.631146 | 2,992,499,459 | 143.640 | 83.790 | 1.616e-06 |

The latency-dominated region extends through the small sizes. N=57,344 is the exact 56-tile/56-core occupancy knee. Throughput continues rising after full occupancy and reaches 2.99 billion qmul/s at N=1,048,576, so this sweep establishes a broadening plateau rather than a hard saturation point.
