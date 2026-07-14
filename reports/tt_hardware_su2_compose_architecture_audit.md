# SU2ComposeBench architecture and eligibility audit

Audit date: `2026-07-14`
Execution source: `cec33db64f9b3acd1264018d85f557d164e04551`
Candidate SHA-256: `2f783b375b7a7f206f7d96db9dd38c9d5d736da27a1cf68e1ea1ea3e9ac16197`
Pinned TT-Metal: `dd2849b5bc6b7a5d38a9eafbeba31ef8d530f8d4`

## Result

The initial H1 candidate passed its preregistered device-0 conformance gates.
It remains `performance_eligible=false` in this report. The eligibility change
must be a separate source commit and rebuild.

## Structural audit

- The candidate creates exactly one unit mesh for device 0 and closes it once.
- Device 1 is not opened or used.
- Host inputs use step-major, six-plane Float32 32x32 tiles.
- Work is split row-major across `min(ceil(B/1024), 56)` cores.
- B=32 selected one core; B=2048 selected two cores.
- Reader and writer kernels contain tile DMA and synchronization only.
- Quaternion Hamilton products and complex-phase products are implemented in
  `su2_compute_common.h` through SFPU multiply/add/subtract helpers.
- The fused path performs one workload dispatch per chain.
- Fused accumulators alternate between six-plane L1 circular-buffer banks;
  only a distinct final-output bank is visible to the writer.
- The unfused path performs exactly `K-1` dispatches and alternates two DRAM
  accumulator buffers. Runtime arguments change between enqueues without a
  program rebuild.

## Conformance evidence

| B | K | cores | values per path | fused max error | unfused max error | CPU oracle error |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 8 | 1 | 192 | 1.417e-7 | 1.417e-7 | 1.794e-15 |
| 2048 | 8 | 2 | 12288 | 2.969e-7 | 2.969e-7 | 2.109e-15 |

Both paths recorded zero failing values and zero nonfinite values. Whole-output
acceptance used `atol=1e-4`, `rtol=1e-4` against Float64 composition of the
exact serialized Float32 inputs. The independent complex128 matrix and Float64
quaternion/phase oracles agreed within the preregistered `1e-11` gate.

## Nonclaims

This audit is silicon conformance, not performance qualification evidence. It
makes no stability, acceleration, CPU-comparison, bandwidth, energy,
dual-device, or full device-side Hamiltonian-lowering claim.
