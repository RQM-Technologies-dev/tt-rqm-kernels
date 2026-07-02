# GitHub Discussion Draft

Title:

```text
Placement guidance for external structured quaternion kernel benchmark
```

Body:

```text
Hi Tenstorrent maintainers,

RQM Technologies has started an independent open-source project called tt-rqm-kernels:

https://github.com/RQM-Technologies-dev/tt-rqm-kernels

The project defines CPU/PyTorch reference kernels and a benchmark suite called StructuredBench for structured tensor operators where quaternion, rotor, phase, and orientation state is represented inside ordinary floating-point tensors.

The first layout is:

[N, 4] = [real, i, j, k]

The first proposed kernel is qmul, the Hamilton product for two [N, 4] inputs producing one [N, 4] output. The second proposed target is qrotate_vector, a streamed rotor/vector rotation built from two Hamilton products.

We have a CPU/PyTorch reference benchmark and want guidance on the right first Tenstorrent integration point. Should this begin as a TT-Metalium programming example, a TT-NN custom operator, or another path?

The benchmark now reports throughput, latency, numerical error, estimated FLOPs/sec, effective GB/sec, arithmetic intensity, and scalar-reference spot-check error. The intent is to make future TT-Metalium or TT-NN results directly comparable against the CPU/PyTorch reference.

Relevant docs:

- docs/structuredbench-spec.md
- docs/operator-contracts.md
- docs/tenstorrent-rfc.md

This is not a request for native quaternion hardware or a new chip feature. The goal is a small, reproducible structured-kernel workload that is below applications and above scalar elementwise math.

Any guidance on preferred placement, benchmark format, or custom-op pathway would be appreciated.
```
