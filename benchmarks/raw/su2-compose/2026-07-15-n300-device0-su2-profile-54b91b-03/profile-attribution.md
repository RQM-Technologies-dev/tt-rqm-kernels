# SU2ComposeBench profiler attribution

Diagnostic Device Program Profiler and Tracy evidence for the exact retained `54b91b…` candidate. Individual captures and this aggregate remain `stable_benchmark=false`.

| case | cores | fused ms | unfused ms | fused device role | unfused device role | dispatch + finish / timed pair |
|---|---:|---:|---:|---|---|---:|
| B=32,768, K=8 | 32 | 32.613 | 34.285 | writer | writer | 0.168 |
| B=8,192, K=32 | 8 | 40.013 | 39.721 | writer | writer | 0.152 |
| B=512, K=512 | 1 | 43.442 | 62.999 | writer | writer | 0.369 |
| B=65,536, K=128 | 56 | 41.387 | 55.083 | writer | writer | 0.239 |

The most frequent fused critical device role is `writer`. Reader, compute, and writer KERNEL scopes are reported separately in the machine-readable artifact.

Reader, compute, and writer scopes overlap in every fused dispatch, and the marginally longest role is less than five percent beyond the next-longest role in every case. No isolated architectural correction is supported, so the exact `54b91b…` candidate is retained for the new stability contract.

The pinned profiler does not expose direct circular-buffer wait or SFPU-utilization counters. Their absence is not interpreted as zero wait or full utilization.

No stability, acceleration, CPU-speedup, measured-bandwidth, or application claim is made.
