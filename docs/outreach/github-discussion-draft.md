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

The first proposed kernel is qmul, the Hamilton product for two [N, 4] inputs producing one [N, 4] output.

Quaternions are useful here because they give us a compact four-lane way to carry rotation, phase, orientation, and related geometric state inside ordinary floating-point tensors.

Where should a minimal TT-Metalium qmul example for [N, 4] structured tensors live?

Secondary questions:

- If a TT-Metalium programming example is not the right starting point, is there a preferred TT-NN custom-op path?
- Would a TT-MLIR representation be useful later, after there is a concrete lower-stack qmul example?

The benchmark now reports throughput, latency, numerical error, estimated FLOPs/sec, effective GB/sec, arithmetic intensity, and scalar-reference spot-check error. The intent is to make future TT-Metalium or TT-NN results directly comparable against the CPU/PyTorch reference.

Relevant docs:

- docs/structuredbench-spec.md
- docs/operator-contracts.md
- docs/tenstorrent-rfc.md

This is not a request for native quaternion hardware or a new chip feature. The goal is a small, reproducible structured-kernel workload that is below applications and above scalar elementwise math.

Any guidance on preferred placement, benchmark format, or custom-op pathway would be appreciated.
```
