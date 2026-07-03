# tt-rqm-kernels Collaboration Plan

## Partnership Frame

RQM is not asking Tenstorrent to support quaternion math as a special feature.

The collaboration frame is:

> RQM is building a structured-kernel benchmark and operator family that helps Tenstorrent demonstrate accelerator strength beyond LLM inference: rotation, phase, orientation, geometry, wave state, and scientific kernels represented inside ordinary floating-point tensors.

`tt-rqm-kernels` should be positioned as lower-stack structured-compute infrastructure: kernels, benchmarks, compiler-lowering questions, examples, and developer education.

Defense remains a downstream application area, not the public lead.

## Current State

The repo is ready for a first handshake:

- independent RQM Technologies project, not an official Tenstorrent repo
- CPU/PyTorch reference kernels for quaternion and rotor operators
- scalar reference checks for independent correctness spot checks
- StructuredBench benchmark reports with latency, throughput, numerical error, estimated FLOPs/sec, effective GB/sec, and arithmetic intensity
- Tenstorrent-facing docs, operator contracts, outreach packet, and CI
- a GitHub Discussion opened in `tenstorrent/tt-metal`

The next move should be broader than asking for help porting `qmul`. The repo should become a structured-compute collaboration surface.

## Priority Lanes

| Priority | Lane | Goal | Success condition |
| ---: | --- | --- | --- |
| 1 | TT-Lang `qmul` prototype | Fastest bridge from PyTorch reference to Tenstorrent-style custom operation logic | `qmul` runs in TT-Lang simulation and produces a StructuredBench-compatible report |
| 2 | `tt-awesome` entry | Low-friction ecosystem visibility | PR or submission adds `tt-rqm-kernels` as a structured-kernel benchmark project |
| 3 | TT-Metalium `qmul` example | Prove RQM can operate at the lower stack | Minimal `[N, 4]` `qmul` kernel compared against CPU/PyTorch and scalar references |
| 4 | StructuredBench report standard | Make this useful as a reusable benchmark class | CPU, TT-Lang, and future TT-Metalium reports share `structuredbench.v1` fields |
| 5 | TT-NN wrapper | Make kernels usable by ordinary Tenstorrent developers | `qmul` or `qrotate_vector` exposed through a TT-NN-style wrapper after lower-stack proof |
| 6 | TT-MLIR lowering discussion | Explore compiler value after a working kernel exists | Concrete question: should `qmul` lower as a fused kernel instead of scalar expansion? |
| 7 | Developer tutorial or blog | Make RQM useful to the ecosystem | Public tutorial explains structured `[N, 4]` kernels and the benchmark path |
| 8 | Cloud/hardware validation | Turn the benchmark into performance evidence | First Tenstorrent hardware report compares CPU/PyTorch vs Tenstorrent backend |

## Next Repo Work

### 1. Add `docs/collaboration-map.md`

Purpose:

- explain how RQM can collaborate across TT-Lang, TT-Metalium, TT-NN, TT-MLIR, `tt-awesome`, and Tenstorrent Cloud
- keep the public frame as structured computation on open accelerators
- avoid pitching native quaternion hardware or new silicon features

Content outline:

- current repo assets
- why StructuredBench is the reusable asset
- collaboration lanes and sequence
- what RQM is asking for now
- what RQM is not asking for

### 2. Add `backends/tt_lang/README.md`

Purpose:

- document the planned TT-Lang backend without pretending it already exists
- make simulator-first work explicit

Content outline:

- goal: implement `qmul` in TT-Lang simulation
- input/output contract: `[N, 4] -> [N, 4]`
- validation: compare against CPU/PyTorch and scalar reference
- report target: `structuredbench.v1`
- non-goals: no fake hardware results, no TT-Metalium replacement claim

### 3. Add `docs/tt-lang-qmul-plan.md`

Purpose:

- turn the TT-Lang prototype into a concrete engineering task

Planned sequence:

```text
PyTorch reference qmul
-> TT-Lang simulated qmul
-> TT-Metalium qmul
-> TT-NN wrapper
-> TT-MLIR lowering discussion
```

Minimum acceptance criteria:

- deterministic sample inputs
- TT-Lang simulation output
- CPU/PyTorch comparison
- scalar reference spot check
- JSON report compatible with `structuredbench.v1`
- clear note that results are simulation/reference outputs, not hardware claims

### 4. Add Tracking Issues

Create issues in the `tt-rqm-kernels` repo:

1. `Implement qmul in TT-Lang simulator`
2. `Prepare tt-awesome submission`
3. `Run StructuredBench on Tenstorrent Cloud`
4. `Design minimal TT-Metalium qmul example`
5. `Define TT-NN wrapper path after lower-stack qmul proof`
6. `Draft structured-kernel tutorial for Tenstorrent developers`

Each issue should include:

- goal
- acceptance criteria
- references to relevant docs
- clear non-goals

## Technical Roadmap

### Phase 1: TT-Lang Simulation

Goal:

- prove that `qmul` can be expressed in a Tenstorrent-adjacent custom-operation flow without requiring hardware access first

Tasks:

- research TT-Lang install and simulator entrypoints
- add a `backends/tt_lang/` planning folder
- implement only after the simulator workflow is understood
- emit a StructuredBench-compatible report

Exit criteria:

- `qmul` simulated output matches CPU/PyTorch and scalar reference within documented tolerance
- report states clearly that it is TT-Lang simulation, not hardware performance

### Phase 2: Ecosystem Visibility

Goal:

- make `tt-rqm-kernels` discoverable inside the Tenstorrent ecosystem

Tasks:

- prepare a `tt-awesome` submission
- categorize as structured tensor kernels, custom kernels, scientific/signal workloads, or low-level examples depending on repo taxonomy
- link to README, StructuredBench spec, operator contracts, and outreach packet

Exit criteria:

- submission or PR opened
- public description leads with structured-kernel benchmarking, not quaternion theory

### Phase 3: TT-Metalium `qmul`

Goal:

- create the first real lower-stack kernel target

Tasks:

- follow maintainer guidance from the GitHub Discussion
- start with `[N, 4]` layout
- compare against CPU/PyTorch and scalar references
- report latency, throughput, numerical error, estimated FLOPs/sec, effective GB/sec, and arithmetic intensity

Exit criteria:

- minimal example or external prototype runs
- result is reproducible
- no unsupported claims about Tenstorrent performance

### Phase 4: TT-NN Wrapper

Goal:

- make structured kernels usable from the higher-level Tenstorrent operator layer

Tasks:

- wait until TT-Lang or TT-Metalium proof exists
- follow Tenstorrent's custom-op conventions
- define a golden/reference function
- expose a Python-facing path only if maintainers agree with placement

Exit criteria:

- wrapper has a reference comparison and a clean example
- docs explain when to use the wrapper versus raw benchmark scripts

### Phase 5: TT-MLIR Lowering Discussion

Goal:

- evaluate whether structured operators should lower as fused kernels instead of scalar expansions

Precondition:

- working `qmul` backend and at least one higher-level wrapper or integration sketch

Core question:

```text
Should qmul lower as a fused structured operator rather than expanding into scalar multiply/add lanes?
```

Exit criteria:

- compiler discussion is grounded in working backend evidence, not speculation

### Phase 6: Hardware Report

Goal:

- publish the first real Tenstorrent hardware comparison

Path:

```text
PyTorch correctness
-> TT-Lang simulation
-> tt-emule no-hardware validation
-> Tenstorrent Cloud hardware result
```

Exit criteria:

- public report distinguishes CPU reference, simulation, emulation, and hardware
- report includes methodology and exact commands
- report avoids claiming stable hardware performance from sample outputs

## Outreach Sequence

1. Continue the `tenstorrent/tt-metal` GitHub Discussion.
2. Post a short Discord note pointing to that Discussion.
3. Prepare `tt-awesome` submission.
4. If there is no Discussion reply after about one week, open a shorter `tt-metal` issue:

```text
Feature proposal: minimal TT-Metalium qmul example for structured [N, 4] tensors
```

5. Contact Tenstorrent OSPO only after the repo has a technical packet plus either a TT-Lang simulation or a clear maintainer response.

## Public Messaging Rules

Use this framing:

> Structured computation on open accelerators.

Lead with:

- structured 4-lane tensor values
- compact benchmark between scalar elementwise ops and matmul
- robotics, graphics, wireless, imaging, wave simulation, physical AI, scientific computing, and signal processing
- defense only as downstream

Avoid:

- asking for native quaternion hardware
- implying Tenstorrent endorsement
- speculative physics claims
- claiming hardware performance from CPU/PyTorch sample reports
- presenting TT-NN or TT-MLIR as first-step asks before lower-stack evidence exists

## Reference Links

- Tenstorrent software stack: https://docs.tenstorrent.com/getting-started/tt-software-stack.html
- TT-Metalium getting started: https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/get_started/get_started.html
- TT-Lang overview: https://docs.tenstorrent.com/tt-lang/overview.html
- TT-Lang repo: https://github.com/tenstorrent/tt-lang
- tt-awesome repo: https://github.com/tenstorrent/tt-awesome
- TT-NN docs: https://docs.tenstorrent.com/tt-metal/latest/ttnn/
- tt-emule repo: https://github.com/tenstorrent/tt-emule
- Tenstorrent Cloud: https://tenstorrent.com/en/hardware/cloud
- TT-MLIR repo: https://github.com/tenstorrent/tt-mlir
- TT-Metalium support: https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/resources/support.html
- Tenstorrent GitHub org: https://github.com/tenstorrent
