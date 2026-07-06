# tt-rqm-kernels

Structured quaternion and rotor kernels for Tenstorrent hardware

[![CI](https://github.com/RQM-Technologies-dev/tt-rqm-kernels/actions/workflows/ci.yml/badge.svg)](https://github.com/RQM-Technologies-dev/tt-rqm-kernels/actions/workflows/ci.yml)

`tt-rqm-kernels` is an independent open-source "RQM Technologies" project for structured numerical kernels targeting the Tenstorrent ecosystem. The first release surface is a correctness-tested CPU/PyTorch reference library. Later work can move selected kernels down into TT-Metalium, TT-NN, and TT-Forge / TT-MLIR.

This is not an official Tenstorrent repository unless and until accepted or co-developed by Tenstorrent.

## Tenstorrent Quick Read

For Tenstorrent engineers:

1. What this is: structured `[N, 4]` float tensor kernels, starting with `qmul`.
2. Why it matters: compact benchmark between scalar elementwise ops and matmul.
3. Current evidence: CPU/PyTorch reference, TT-Lang simulator, and tt-emule candidate.
4. Current ask: enable or run one hardware-labeled `qmul` report.
5. Best next link: [docs/tenstorrent-engineer-copy-paste-packet.md](docs/tenstorrent-engineer-copy-paste-packet.md).

The single next action is to help produce one hardware-labeled StructuredBench
`qmul` report. Placement guidance is welcome if it arrives, but the active plan
no longer waits on it.

## For Tenstorrent Reviewers

1. Start here: [docs/tenstorrent-landing.md](docs/tenstorrent-landing.md)
2. Technical spec: [docs/structuredbench-spec.md](docs/structuredbench-spec.md)
3. First ask: [docs/tt-metalium-qmul-design.md](docs/tt-metalium-qmul-design.md)
4. Emulation evidence: [docs/tt-emule-qmul-validation-plan.md](docs/tt-emule-qmul-validation-plan.md)
5. Outreach packet: [reports/tenstorrent_packet.md](reports/tenstorrent_packet.md)
6. Current status command: `python scripts/repo_status.py`

Current blocker: the experimental TT-Metalium scalar RISC-V `qmul` candidate
has now built and produced an emulation-labeled StructuredBench report through
`tt-emule`. It is not a hardware result. The next implementation step is a real
Tenstorrent Cloud or hardware run through the existing external candidate path.

## Functional Tenstorrent Path

The current functional path is the StructuredBench `external-qmul` protocol:
CPU/PyTorch reference validation, optional tt-emule execution through the
experimental TT-Metalium candidate wrapper, and a hardware mode that requires a
real Tenstorrent Cloud or device command before it can run.

Start here:

- [docs/tenstorrent-functional-quickstart.md](docs/tenstorrent-functional-quickstart.md)
- `python scripts/rqm_tt_quickstart.py --check`
- Browser silicon target: Tenstorrent Cloud Console
- [docs/tenstorrent-console-access-plan.md](docs/tenstorrent-console-access-plan.md)
- [docs/tenstorrent-console-copy-paste.md](docs/tenstorrent-console-copy-paste.md)
- `python scripts/rqm_tt_console_runner.py --check`
- [docs/outreach/tenstorrent-console-access-request.md](docs/outreach/tenstorrent-console-access-request.md)
- [docs/tenstorrent-cloud-access-plan.md](docs/tenstorrent-cloud-access-plan.md)
- [docs/tenstorrent-engineer-copy-paste-packet.md](docs/tenstorrent-engineer-copy-paste-packet.md)
- `python scripts/rqm_tt_cloud_runner.py --check`
- [examples/tenstorrent_qmul_quickstart.py](examples/tenstorrent_qmul_quickstart.py)
- [docs/tenstorrent-value-proposition.md](docs/tenstorrent-value-proposition.md)

For no-cost hardware validation, prefer delegated Tenstorrent-run validation or
explicitly granted no-cost Tenstorrent Cloud access. This repo does not
implement payment-backed cloud provisioning, cloud billing integration, or
credential storage.

Observed Console fit: API inference, Usage, Billing, Compute, and Resources are
visible; no dedicated hardware allocation is assumed; Instances and Baremetal
are treated as blocked until access is granted. The hardware request path is
`Compute -> Resources -> Request Capacity` for one small `[N, 4]`
StructuredBench `qmul` report, then either a VSCode/browser instance run or SSH
baremetal run. If the Console form has no selectable Resource Type, use the
Tenstorrent engineer packet for delegated validation. Tenstorrent support has
acknowledged request `CUST-812` for TT-Cloud access for this StructuredBench
`qmul` hardware validation.

## Core Idea

RQM Technologies develops structured numerical kernels where quaternions, rotors, phase, orientation, and wave states are represented inside ordinary floating-point tensors.

A quaternion can live inside floats as:

```text
[..., 4] = [real, i, j, k]
```

The tensor is still a regular real-valued PyTorch tensor. The structure comes from the convention and the operators applied to the final dimension. For example, a quaternion multiplication kernel consumes two tensors with final dimension `4` and returns a tensor with the same final dimension.

This keeps the data layout friendly to accelerator stacks while preserving useful algebraic structure at the operator level.

## Why Structured Tensor Kernels Matter

Modern AI accelerators are optimized around dense tensor movement, tiling, and fused math. Many domains, however, carry structured state:

- orientations and poses in robotics
- rotations and streams of transforms in graphics
- phase in wireless and signal processing
- vector and wave state in imaging and simulation
- geometric features in scientific computing and physical AI
- downstream signals processing where these numerical patterns are relevant

Representing these states as ordinary floating-point tensors makes them compatible with accelerator data paths. Implementing the right structured kernels then lets software keep the math meaningful without leaving the tensor runtime.

The goal of this repository is correctness first:

- simple PyTorch reference operators
- clear validation and broadcasting rules
- tests for algebraic identities and numerical tolerances
- benchmarks that make future accelerator ports comparable
- StructuredBench reports for quaternion, rotor, inverse, normalization, and phase workloads

## Current Milestone

Phase 1 implements CPU/PyTorch reference kernels for quaternion and rotor operations:

- Hamilton product: `qmul`
- conjugate, norm, normalization, inverse, and dot product
- vector rotation by unit rotors
- phase and orientation tracking helpers
- examples, benchmark scripts, and the StructuredBench CLI

Run the benchmark smoke suite:

```bash
python -m tt_rqm_kernels.structuredbench --suite smoke
```

Run focused or full suites:

```bash
python -m tt_rqm_kernels.structuredbench --suite qmul
python -m tt_rqm_kernels.structuredbench --suite qrotate
python -m tt_rqm_kernels.structuredbench --suite full
```

StructuredBench emits a versioned report schema intended to compare CPU/PyTorch reference results against later TT-Metalium and TT-NN backend implementations.

## Optional TT-Lang Simulator qmul Smoke

The first Tenstorrent-adjacent prototype is an optional TT-Lang functional
simulator smoke for `qmul` over `[N, 4]` row-major float32 tensors. It is not a
hardware backend and does not claim hardware performance.

Check whether the simulator is available:

```bash
python scripts/run_ttlang_qmul_smoke.py --check
```

The check output also reports whether `tt-lang-sim-stats` is available. Stats
support is optional; missing stats tooling does not block simulator correctness
runs.

Run it in an environment with `tt-lang-sim` installed:

```bash
python scripts/run_ttlang_qmul_smoke.py \
  --items 128 \
  --json-output reports/tt_lang_qmul_sim.json \
  --markdown-output reports/tt_lang_qmul_sim.md
```

The default TT-Lang simulator variant is `block`, which splits each `[N, 4]`
quaternion into lane slices inside TT-Lang dataflow buffers. An experimental
`raw` variant uses TT-Lang raw element reads and writes to make the four scalar
lane equations explicit inside the simulator:

```bash
python scripts/run_ttlang_qmul_smoke.py \
  --variant raw \
  --items 128 \
  --iters 1 \
  --warmup 0
```

Both variants are simulator-only checks. They compare implementation shape and
trace/stat diagnostics, not hardware performance.

Optionally capture a simulator trace and post-process it with
`tt-lang-sim-stats` when that tool is installed:

```bash
python scripts/run_ttlang_qmul_smoke.py \
  --items 128 \
  --iters 1 \
  --warmup 0 \
  --seed 0 \
  --trace-output reports/tt_lang_qmul_trace.jsonl \
  --stats-output reports/tt_lang_qmul_stats.txt \
  --json-output reports/tt_lang_qmul_sim.json \
  --markdown-output reports/tt_lang_qmul_sim.md
```

Or run the same optional backend through StructuredBench:

```bash
python -m tt_rqm_kernels.structuredbench \
  --backend tt-lang-sim \
  --suite qmul \
  --items 128 \
  --iters 1 \
  --warmup 0
```

StructuredBench exposes the same trace/stat capture through TT-Lang-specific
flags:

```bash
python -m tt_rqm_kernels.structuredbench \
  --backend tt-lang-sim \
  --suite qmul \
  --tt-lang-variant raw \
  --items 128 \
  --iters 1 \
  --warmup 0 \
  --tt-lang-trace-output reports/tt_lang_qmul_trace.jsonl \
  --tt-lang-stats-output reports/tt_lang_qmul_stats.txt
```

Trace/stat output is simulator diagnostics only. It is useful for inspecting
dataflow-buffer and copy activity before a hardware backend exists, but it is
not hardware performance evidence.

See [docs/tt-lang-qmul-plan.md](docs/tt-lang-qmul-plan.md) for setup details
and acceptance criteria.

## Why Tenstorrent Developers Should Care

This repo gives Tenstorrent a compact benchmark class between scalar elementwise ops and large matmul. Some workloads need to preserve structure inside the data: rotation, phase, orientation, direction, and geometric state. That shows up in robotics pose updates, graphics rotation streams, wireless phase tracking, imaging, wave simulation, physical AI, scientific computing, signal processing, and downstream signals processing.

The first benchmark target is `qmul` over `[N, 4]` floating-point tensors. `qmul` is small enough to validate, but structured enough to test cross-lane dependencies, fixed multiply/add/sign patterns, data movement, fusion, register reuse, and arithmetic intensity. This can help Tenstorrent show useful accelerator behavior beyond LLM inference and matmul-heavy neural networks.

No native quaternion datatype, new silicon feature, or hardware change is required. The values stay inside ordinary floating-point tensors.

Simple proof path:

```text
CPU/PyTorch qmul reference
-> scalar correctness check
-> TT-Lang simulator qmul for [N, 4]
-> tt-emule run of real TT-Metalium qmul candidate
-> real TT-Metalium / Tenstorrent hardware report
-> compare throughput, latency, numerical error, FLOPs/sec, GB/sec, and arithmetic intensity
```

## Long-Term QuantumIR Direction For Classical AI Compute

QuantumIR here means a classical/AI accelerator front end for selected
quantum-mechanics workloads, not a quantum-hardware proposal. The long-term
direction is to lower selected workloads into the structured quaternion, rotor,
phase, and tensor kernels already defined here. The near-term target is not a
new backend or a claim about replacing quantum hardware. It is a documentation
and validation path for representing small SU(2)-style operations as ordinary
floating-point tensor kernels.

The first proposed QuantumIR target is a single-qubit SU(2) gate represented as
a unit quaternion rotor and validated against a standard 2x2 complex matrix
reference. That layer should lower into the existing `[N, 4]` kernel foundation
and StructuredBench report path rather than displacing the active qmul,
tt-emule, Cloud/hardware, or maintainer-placement work.

See:

- [docs/quantum-ir.md](docs/quantum-ir.md)
- [docs/quantum-ir-roadmap.md](docs/quantum-ir-roadmap.md)
- [docs/quantum-ir-operator-mapping.md](docs/quantum-ir-operator-mapping.md)

## StructuredBench Hardware Metrics

StructuredBench now reports hardware-relevant estimates alongside latency, throughput, and numerical error:

- estimated FLOPs
- estimated FLOPs/sec
- estimated bytes read and written
- effective GB/sec
- arithmetic intensity in FLOPs/byte

The estimates are intentionally simple and documented. For example, `qmul` counts 28 FLOPs per Hamilton product, reads two 4-lane quaternion inputs, and writes one 4-lane quaternion output. These are comparison metrics for backend evaluation, not hardware-counter measurements.

Committed reports are sample CPU/PyTorch reference outputs. They are included to show the report shape and outreach packet format, not to claim stable hardware performance.

Generate JSON and Markdown reports:

```bash
python -m tt_rqm_kernels.structuredbench \
  --suite smoke \
  --json-output reports/structuredbench_latest.json \
  --markdown-output reports/structuredbench_latest.md
```

## Connection Points To Tenstorrent

The Tenstorrent-facing surfaces are:

- [docs/tenstorrent-landing.md](docs/tenstorrent-landing.md)
- [docs/tenstorrent-rfc.md](docs/tenstorrent-rfc.md)
- [docs/collaboration-map.md](docs/collaboration-map.md)
- [docs/structuredbench-spec.md](docs/structuredbench-spec.md)
- [docs/structured-qmul-tutorial.md](docs/structured-qmul-tutorial.md)
- [docs/tenstorrent-execution-runbook.md](docs/tenstorrent-execution-runbook.md)
- [docs/tenstorrent-functional-quickstart.md](docs/tenstorrent-functional-quickstart.md)
- [docs/tenstorrent-console-access-plan.md](docs/tenstorrent-console-access-plan.md)
- [docs/tenstorrent-console-copy-paste.md](docs/tenstorrent-console-copy-paste.md)
- [docs/tenstorrent-cloud-access-plan.md](docs/tenstorrent-cloud-access-plan.md)
- [docs/tenstorrent-engineer-copy-paste-packet.md](docs/tenstorrent-engineer-copy-paste-packet.md)
- [docs/tenstorrent-hardware-command-contract.md](docs/tenstorrent-hardware-command-contract.md)
- [docs/tenstorrent-delegated-validation.md](docs/tenstorrent-delegated-validation.md)
- [docs/tenstorrent-value-proposition.md](docs/tenstorrent-value-proposition.md)
- [docs/tt-emule-qmul-validation-plan.md](docs/tt-emule-qmul-validation-plan.md)
- [docs/complex-quaternion-bridge.md](docs/complex-quaternion-bridge.md)
- [docs/phase-update-tenstorrent-plan.md](docs/phase-update-tenstorrent-plan.md)
- [docs/external-tenstorrent-contribution-selection.md](docs/external-tenstorrent-contribution-selection.md)
- [docs/physical-ai-pose-stream-demo.md](docs/physical-ai-pose-stream-demo.md)
- [docs/structuredbench-hpc-expansion-roadmap.md](docs/structuredbench-hpc-expansion-roadmap.md)
- [docs/quantum-ir.md](docs/quantum-ir.md)
- [docs/quantum-ir-roadmap.md](docs/quantum-ir-roadmap.md)
- [docs/quantum-ir-operator-mapping.md](docs/quantum-ir-operator-mapping.md)
- [docs/tt-mlir-fused-lowering-prerequisites.md](docs/tt-mlir-fused-lowering-prerequisites.md)
- [docs/operator-contracts.md](docs/operator-contracts.md)
- [docs/tt-lang-qmul-plan.md](docs/tt-lang-qmul-plan.md)
- [docs/structuredbench-opportunity-plan.md](docs/structuredbench-opportunity-plan.md)
- [reports/tenstorrent_packet.md](reports/tenstorrent_packet.md)
- [reports/tenstorrent_hardware_report_template.md](reports/tenstorrent_hardware_report_template.md)
- [reports/tt_emule_qmul_candidate.md](reports/tt_emule_qmul_candidate.md)

Proposed backend path:

- optional TT-Lang simulator `qmul` for `[N, 4]` quaternion tensors
- tt-emule validation for the experimental TT-Metalium `qmul` candidate
- future TT-Metalium `qmul` for `[N, 4]` quaternion tensors
- CPU/PyTorch physical-AI pose stream demo using `qrotate_vector`
- future ComplexTensor-to-QuaternionTensor bridge experiments after lower-stack
  evidence is clearer
- future `phase_update` backend plan after the `qmul` candidate path is stable
- future TT-NN wrapper once lower-stack evidence is reproducible
- future TT-MLIR lowering discussion after an explicit lower-stack kernel exists

## Install for Development

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

## Repository Layout

```text
tt_rqm_kernels/        Reference Python package
docs/                  Thesis and Tenstorrent backend roadmap
examples/              Small domain-oriented usage examples
benchmarks/            CPU reference benchmark scripts
reports/               Generated StructuredBench and outreach reports
tests/                 Correctness and shape tests
```

See [docs/structuredbench.md](docs/structuredbench.md) for benchmark suite details.

## License

Apache License 2.0. See `LICENSE`.
