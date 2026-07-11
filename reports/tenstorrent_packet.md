# Tenstorrent Outreach Packet

## Project Summary

`tt-rqm-kernels` is an independent RQM Technologies LLC project for structured quaternion, rotor, and phase-aware tensor kernels represented inside ordinary floating-point tensors. StructuredBench provides a conformance-gated benchmark contract and an implemented scalar RISC-V TT-Metalium correctness baseline.

Committed reports are sample CPU/PyTorch reference outputs. They are included to show the report shape and outreach packet format, not to claim stable hardware performance.

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
CPU/PyTorch qmul reference
-> scalar correctness check
-> TT-Lang simulator qmul for [N, 4]
-> tt-emule run of real TT-Metalium qmul candidate
-> real TT-Metalium / Tenstorrent hardware report
-> compare throughput, latency, numerical error, FLOPs/sec, GB/sec, and arithmetic intensity
```

## Immediate Ask

The current request is one Stage A silicon-conformance `qmul` run using the delegated engineer packet: [docs/tenstorrent-engineer-copy-paste-packet.md](../docs/tenstorrent-engineer-copy-paste-packet.md).

The returned artifacts should be:

```text
reports/tt_hardware_qmul_quickstart.json
reports/tt_hardware_qmul_quickstart.md
reports/tt_hardware_qmul_environment.txt
```

Use `execution_label=hardware`, `benchmark_stage=conformance`, and `stable_benchmark=false` only for real Tenstorrent hardware.

## Long-Term Direction: QuantumIR for Classical AI Compute

QuantumIR here means a classical/AI accelerator front end for selected quantum-mechanics workloads, not a quantum-hardware proposal. The immediate ask remains narrow: one Stage A silicon conformance run for the existing `[N, 4]` `qmul` candidate.

Longer term, RQM Technologies is exploring QuantumIR as a domain-facing layer above these kernels. It would lower selected quantum-mechanics workloads on classical Tenstorrent/AI accelerators, including SU(2) rotations, unitary composition, Hamiltonian evolution, phase/coherence updates, and AI augmentation use cases, into the same structured quaternion, rotor, phase, and tensor operators used by StructuredBench.

This does not claim that arbitrary quantum computation is efficiently classically simulable, does not ask Tenstorrent for native quaternion hardware, and does not replace the signal processing, physical AI, imaging, wave simulation, and scientific computing kernel story. It is a future front end built on the same kernel foundation.

## Benchmark Table

| workload | items | iters | latency_ms | throughput | unit | max_abs_err |
| --- | --- | --- | --- | --- | --- | --- |
| qmul | 1024 | 5 | 0.1719 | 5955884.68 | qmul/s | 1.179e-07 |
| qrotate | 1024 | 5 | 0.4109 | 2491841.90 | rotations/s | 4.148e-07 |
| qnormalize | 1024 | 5 | 0.0432 | 23694159.54 | normalizations/s | 1.014e-07 |
| qinverse | 1024 | 5 | 0.2534 | 4040758.04 | inverses/s | 1.486e-06 |
| phase_update | 2048 | 5 | 0.1506 | 13595218.36 | phase-updates/s | 3.465e-07 |

## Hardware Metrics Table

| workload | items | estimated_flops | estimated_flops_per_s | estimated_total_bytes | effective_gb_per_s | arithmetic_intensity |
| --- | --- | --- | --- | --- | --- | --- |
| qmul | 1024 | 143360 | 1.668e+08 | 245760 | 0.286 | 0.583 |
| qrotate | 1024 | 327680 | 1.595e+08 | 204800 | 0.100 | 1.600 |
| qnormalize | 1024 | 66560 | 3.080e+08 | 163840 | 0.758 | 0.406 |
| qinverse | 1024 | 76800 | 6.061e+07 | 163840 | 0.129 | 0.469 |
| phase_update | 2048 | 61440 | 8.157e+07 | 204800 | 0.272 | 0.300 |

## Proposed First TT-Metalium Target

Implemented Stage A TT-Metalium target: scalar RISC-V `qmul` for `[N, 4]` quaternion tensors; not performance-eligible.

## Proposed Second Target

Proposed second target: `qrotate_vector` for streamed unit-rotor/vector rotation.

## Relevant Docs

- [docs/tenstorrent-landing.md](../docs/tenstorrent-landing.md)
- [docs/tenstorrent-engineer-copy-paste-packet.md](../docs/tenstorrent-engineer-copy-paste-packet.md)
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

RQM Technologies has a CPU/PyTorch reference benchmark for structured quaternion and rotor tensor kernels, with qmul as the proposed first [N, 4] TT-Metalium target.

The repo now has a one-command readiness check (`python scripts/rqm_tt_quickstart.py --check`), an external-qmul protocol for candidate commands, and tt-emule evidence for the experimental TT-Metalium candidate. The tt-emule report is emulation-only and is not hardware performance.

Could Tenstorrent enable or run one Stage A hardware conformance report for the existing [N, 4] TT-Metalium qmul candidate?

Secondary questions, only if there is actionable guidance: where should the existing TT-Metalium qmul example live, and is there a preferred TT-NN custom-op path after Stage B hardware evidence exists?

The benchmark reports throughput, latency, numerical error, estimated FLOPs/sec, effective GB/sec, and arithmetic intensity, with scalar-reference spot checks for correctness.
```

## Suggested Discord Post

```text
Hi Tenstorrent community, RQM Technologies is building an independent structured-kernel benchmark for quaternion and rotor tensor operators represented inside ordinary floating-point tensors.

The immediate ask is one Stage A qmul silicon-conformance report, not a new hardware feature or placement decision. The copy/paste packet is in docs/tenstorrent-engineer-copy-paste-packet.md.

Repo: https://github.com/RQM-Technologies-dev/tt-rqm-kernels
```
