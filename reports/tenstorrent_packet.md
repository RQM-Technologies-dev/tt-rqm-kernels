# Tenstorrent Outreach Packet

## Project Summary

`tt-rqm-kernels` is an independent RQM Technologies LLC project for structured quaternion, rotor, and phase-aware tensor kernels represented inside ordinary floating-point tensors. StructuredBench provides conformance-gated benchmark contracts, CPU/PyTorch references, simulator and emulator paths, and reproducible Wormhole/N300 evidence for `qmul`, fused SU(2) composition, and H2A Hamiltonian lowering.

The table below is generated from the committed CPU/PyTorch reference report. Separate linked artifacts contain real Tenstorrent hardware evidence with their own provenance, claim levels, and limitations.

Report labels:

```text
execution_label: cpu
stable_benchmark: false
methodology_note: CPU/PyTorch reference run; not a hardware performance result.
```

## Why Tenstorrent Developers Should Care

StructuredBench gives Tenstorrent a compact benchmark class between scalar elementwise ops and large matmul. It focuses on structured 4-lane tensor values that carry rotation, phase, orientation, direction, and geometric state inside ordinary floating-point tensors.

The first target is `qmul` over `[N, 4]` tensors. It is small enough to validate with CPU/PyTorch and scalar references, but structured enough to exercise cross-lane dependencies, fixed multiply/add/sign patterns, data movement, fusion, register reuse, and arithmetic intensity. No native quaternion datatype, new silicon feature, or hardware change is required.

Proof path:

```text
CPU/PyTorch qmul reference: complete
-> scalar correctness check: complete
-> TT-Lang simulator and tt-emule TT-Metalium paths: complete
-> N300 Stage A silicon conformance: complete
-> multicore Tensix/SFPU Stage B evidence: complete
-> three-session one-device qmul stability qualification: Claim Level 2
-> current-main upstream-shaped example: correctness validated; placement pending
```

## Immediate Ask

The remaining external request is maintainer placement guidance in [tenstorrent/tt-metal#49887](https://github.com/tenstorrent/tt-metal/issues/49887): should the minimal FP32 `[N,4]` Hamilton-product example live as a contributed TT-Metalium programming example or an experimental TT-NN operation?

A self-contained current-main programming example is available on the public review branch. Its N=128 and N=4096 N300 runs are correctness evidence only; the port does not inherit the protected external candidate's Claim Level 2 status and makes no acceleration claim.

## Long-Term Direction: QuantumIR for Classical AI Compute

QuantumIR here means a classical/AI accelerator front end for selected quantum-mechanics workloads, not a quantum-hardware proposal. The immediate ask remains the narrow placement decision for the existing current-main `[N, 4]` `qmul` example.

Longer term, RQM Technologies is exploring QuantumIR as a domain-facing layer above these kernels. It would lower selected quantum-mechanics workloads on classical Tenstorrent/AI accelerators, including SU(2) rotations, unitary composition, Hamiltonian evolution, phase/coherence updates, and AI augmentation use cases, into the same structured quaternion, rotor, phase, and tensor operators used by StructuredBench.

This does not claim that arbitrary quantum computation is efficiently classically simulable, does not ask Tenstorrent for native quaternion hardware, and does not replace the signal processing, physical AI, imaging, wave simulation, and scientific computing kernel story. It is a future front end built on the same kernel foundation.

## CPU/PyTorch Reference Table

| workload | items | iters | latency_ms | throughput | unit | max_abs_err |
| --- | --- | --- | --- | --- | --- | --- |
| qmul | 1024 | 5 | 0.1719 | 5955884.68 | qmul/s | 1.179e-07 |
| qrotate | 1024 | 5 | 0.4109 | 2491841.90 | rotations/s | 4.148e-07 |
| qnormalize | 1024 | 5 | 0.0432 | 23694159.54 | normalizations/s | 1.014e-07 |
| qinverse | 1024 | 5 | 0.2534 | 4040758.04 | inverses/s | 1.486e-06 |
| phase_update | 2048 | 5 | 0.1506 | 13595218.36 | phase-updates/s | 3.465e-07 |

## Reference Workload-Shape Metrics

| workload | items | estimated_flops | estimated_flops_per_s | estimated_total_bytes | effective_gb_per_s | arithmetic_intensity |
| --- | --- | --- | --- | --- | --- | --- |
| qmul | 1024 | 143360 | 1.668e+08 | 245760 | 0.286 | 0.583 |
| qrotate | 1024 | 327680 | 1.595e+08 | 204800 | 0.100 | 1.600 |
| qnormalize | 1024 | 66560 | 3.080e+08 | 163840 | 0.758 | 0.406 |
| qinverse | 1024 | 76800 | 6.061e+07 | 163840 | 0.129 | 0.469 |
| phase_update | 2048 | 61440 | 8.157e+07 | 204800 | 0.272 | 0.300 |

## Current TT-Metalium Result

The scalar RISC-V Stage A baseline and multicore Tensix/SFPU Stage B candidate both ran on Wormhole. Three qualified device-0 sessions support the protected aggregate qmul Claim Level 2 release; this is stable one-device performance evidence, not a CPU or application acceleration claim.

## Proposed Second Target

Proposed second target: `qrotate_vector` for streamed unit-rotor/vector rotation.

## Relevant Docs

- [docs/tenstorrent-landing.md](../docs/tenstorrent-landing.md)
- [docs/benchmarks/wormhole-qmul.md](../docs/benchmarks/wormhole-qmul.md)
- [docs/benchmarks/wormhole-qmul-hardware-evidence.md](../docs/benchmarks/wormhole-qmul-hardware-evidence.md)
- [docs/upstream/current-main-qmul-port.md](../docs/upstream/current-main-qmul-port.md)
- [docs/operator-contracts.md](../docs/operator-contracts.md)
- [docs/structuredbench-spec.md](../docs/structuredbench-spec.md)
- [docs/tenstorrent-rfc.md](../docs/tenstorrent-rfc.md)
- [reports/tt_emule_qmul_candidate.md](tt_emule_qmul_candidate.md)
- [docs/quantum-ir.md](../docs/quantum-ir.md)
- [docs/quantum-ir-roadmap.md](../docs/quantum-ir-roadmap.md)
- [docs/quantum-ir-operator-mapping.md](../docs/quantum-ir-operator-mapping.md)

## Suggested GitHub Discussion Text

```text
Hi Tenstorrent maintainers,

RQM Technologies maintains an independent structured-kernel benchmark with CPU/PyTorch references and real Wormhole/N300 evidence for FP32 [N,4] qmul.

The external multicore Tensix/SFPU candidate has whole-output validation, profiler diagnostics, and a three-session one-device Claim Level 2 stability qualification. A separate self-contained example was refreshed against tt-metal main and passed N=128 and N=4096 N300 correctness runs; those two runs are correctness-only and do not inherit the protected release status.

Would Tenstorrent prefer the minimal current-main example as a contributed TT-Metalium programming example, or should it remain external / move to an experimental TT-NN ProgramDescriptor operation?

No upstream implementation PR will be opened until maintainers establish the preferred placement and layout boundary.
```

## Suggested Discord Post

```text
Hi Tenstorrent community, RQM Technologies maintains an independent structured-kernel benchmark for quaternion and rotor tensor operators represented inside ordinary floating-point tensors.

The qmul path now has real Wormhole/N300 correctness, profiler, and one-device stability evidence. The remaining ask is placement guidance for the self-contained current-main example; it is not a request for a new datatype, silicon feature, or endorsement.

Repo: https://github.com/RQM-Technologies-dev/tt-rqm-kernels
```
