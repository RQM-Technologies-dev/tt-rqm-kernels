# Wormhole qmul device-1 parity evidence

Diagnostic parity evidence only; this is not device-0 stability, dual-device scaling, aggregate N300 throughput, acceleration, or endorsement.

| N | device 0 median ms | device 1 median ms | D1/D0 | device 0 p95 ms | device 1 p95 ms | cores | correctness |
|---:|---:|---:|---:|---:|---:|---:|---|
| 4,096 | 1.601423 | 1.523639 | 0.9514 | 1.626753 | 1.559684 | 4 | identical |
| 65,536 | 2.077258 | 2.134828 | 1.0277 | 2.146197 | 2.191347 | 56 | identical |
| 262,144 | 4.233331 | 4.288705 | 1.0131 | 4.262865 | 4.301364 | 56 | identical |
