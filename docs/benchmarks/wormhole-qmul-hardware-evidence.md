# Wormhole persistent FP32 qmul hardware evidence

The three stability sessions are published as Claim Level 2. The remaining
measurements on this page are diagnostics, not acceleration, CPU, application,
dual-device, aggregate-N300, hardware-bandwidth, or endorsement claims.

## Outcome

- Three device-0 cold starts passed the fixed Level 2 stability gates. Only the
  qualification artifact says `stable_benchmark=true`; every session remains
  `false`.
- Device 1 passed N=128 conformance and a same-binary three-size parity run.
- Controlled scaling used only physically active cores: 1/2/4 at N=4096 and
  1/2/4/8/16/32/56 at N=65536 and N=262144.
- First-use overhead follows first submission/dispatch initialization, not the
  size placed first in the sequence.
- Device Program Profiler and Tracy captured 56-core N=65536 and N=262144
  execution. Reader, compute, and writer overlap; writer/NCRISC is marginally
  longest and compute is nearly coextensive.
- The nine-size sweep passed through N=1048576. N=57344 is the exact 56-tile,
  56-core occupancy knee; throughput continued rising beyond it.

## Evidence

- [stability qualification](../../benchmarks/processed/wormhole-qmul-stability-qualification.json)
- [device-1 parity](../../benchmarks/processed/wormhole-qmul-device1-parity.md)
- [controlled core scaling](../../benchmarks/processed/wormhole-qmul-core-scaling.md)
- [initialization diagnostics](../../benchmarks/processed/wormhole-qmul-initialization-diagnostics.md)
- [profiler and same-device ceilings](../../benchmarks/processed/wormhole-qmul-profiler-and-ceilings.md)
- [larger-size saturation](../../benchmarks/processed/wormhole-qmul-saturation.md)
- [machine-readable evidence index](../../benchmarks/processed/wormhole-qmul-hardware-evidence-index.json)

Every processed file is deterministically regenerated from the isolated raw
session directories by:

```bash
python scripts/generate_qmul_hardware_evidence.py
```

## Interpretation boundaries

The qmul logical traffic rate continues to mean 48 logical bytes per Hamilton
product. It is not measured DRAM, NoC, or PCIe bandwidth. The pinned TT-Metal
suite supplied a BFP8 FPU matmul as its closest working compute microbenchmark,
not an FP32 SFPU ceiling. The profiler did not expose circular-buffer stall,
NoC-wait, or SFPU-utilization counters. Raw failed microbenchmark attempts are
preserved and called out rather than discarded.
