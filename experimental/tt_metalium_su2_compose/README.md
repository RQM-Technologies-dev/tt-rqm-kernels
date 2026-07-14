# TT-Metalium SU2ComposeBench Candidate

This package contains the separate H1 Wormhole candidate. It consumes
CPU-lowered Float32 rotor and phase chains; it does not lower Hamiltonian
coefficients on device.

The binary implements two paths inside one device-0 session:

- unfused `K-1` qmul-plus-phase dispatches with DRAM ping-pong accumulators;
- one fused reader/compute/writer workload with L1 accumulator ping-pong.

All quaternion and complex-phase arithmetic is in the compute/SFPU kernels.
Data-movement kernels perform only tile DMA and synchronization.

Build against the pinned checkout:

```bash
python experimental/tt_metalium_su2_compose/build_candidate.py \
  --tt-metal-root /path/to/tt-metal \
  --cmake-prefix-path /path/to/tt-metal/build_Release
```

The audited source constant is `performance_eligible=true`. It was promoted
only after both N300 conformance cases passed and the architecture audit was
committed. This architecture qualification does not imply benchmark stability
or an acceleration claim; emitted metrics retain `stable_benchmark=false`.
