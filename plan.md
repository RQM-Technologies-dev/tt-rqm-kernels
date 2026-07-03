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
- optional TT-Lang simulator `qmul` prototype with a StructuredBench-compatible simulator report
- Tenstorrent-facing docs, operator contracts, outreach packet, and CI
- a minimal TT-Metalium `qmul` design document for `[N, 4]` structured tensors
- scientific/HPC positioning that relates RQM structured kernels to
  Tenstorrent's broader non-LLM scientific workload direction
- a GitHub Discussion opened in `tenstorrent/tt-metal`
- a `tt-awesome` submission issue opened:
  https://github.com/tenstorrent/tt-awesome/issues/104

The completed setup work should now be treated as background. The active work is
response tracking, maintainer placement guidance, and the first hardware-facing
implementation path.

## Recommended Next Step

Commit the scientific/HPC positioning wording, then create focused tracking
issues for the remaining collaboration lanes, starting with `tt-awesome`
approval tracking and TT-Metalium `qmul` placement guidance.

Why this is next:

- the repo now has CPU/PyTorch reference results, a TT-Lang simulator proof
  point, and a public collaboration map
- the `tt-awesome` submission is opened and waiting on maintainer-side
  labeling or approval
- the minimal TT-Metalium `qmul` design document is present, so the next step is
  coordination and implementation planning rather than more design prose
- the scientific/HPC positioning now gives RQM a conservative way to reference
  adjacent Tenstorrent scientific workload activity without claiming that
  spectral element methods need quaternion kernels

## Priority Lanes

| Priority | Lane | Goal | Success condition |
| ---: | --- | --- | --- |
| 1 | `tt-awesome` approval | Low-friction ecosystem visibility | Maintainers approve issue #104 and the generated entry PR is merged |
| 2 | TT-Metalium `qmul` example | Prove RQM can operate at the lower stack | Minimal `[N, 4]` `qmul` kernel compared against CPU/PyTorch and scalar references |
| 3 | StructuredBench report standard | Make this useful as a reusable benchmark class | CPU, TT-Lang, and future TT-Metalium reports share `structuredbench.v1` fields |
| 4 | TT-NN wrapper | Make kernels usable by ordinary Tenstorrent developers | `qmul` or `qrotate_vector` exposed through a TT-NN-style wrapper after lower-stack proof |
| 5 | TT-MLIR lowering discussion | Explore compiler value after a working kernel exists | Concrete question: should `qmul` lower as a fused kernel instead of scalar expansion? |
| 6 | Developer tutorial or blog | Make RQM useful to the ecosystem | Public tutorial explains structured `[N, 4]` kernels and the benchmark path |
| 7 | Cloud/hardware validation | Turn the benchmark into performance evidence | First Tenstorrent hardware report compares CPU/PyTorch vs Tenstorrent backend |

## Next Repo Work

### 1. Open Remaining Tracking Issues

Create issues in the `tt-rqm-kernels` repo:

1. `Track tt-awesome submission approval`
2. `Track TT-Metalium qmul placement guidance`
3. `Implement minimal TT-Metalium qmul example after placement is confirmed`
4. `Run StructuredBench on Tenstorrent Cloud`
5. `Define TT-NN wrapper path after lower-stack qmul proof`
6. `Draft structured-kernel tutorial for Tenstorrent developers`

Each issue should include:

- goal
- acceptance criteria
- references to relevant docs
- clear non-goals

## Technical Roadmap

### Phase 1: Ecosystem Visibility

Goal:

- make `tt-rqm-kernels` discoverable inside the Tenstorrent ecosystem

Tasks:

- track `tt-awesome` issue #104 and any generated entry PR
- respond to maintainer placement or metadata feedback
- link to README, StructuredBench spec, operator contracts, and outreach packet

Exit criteria:

- `tt-awesome` approval or generated entry PR merged
- public description leads with structured-kernel benchmarking, not quaternion theory

### Phase 2: TT-Metalium `qmul`

Goal:

- create the first real lower-stack kernel target

Tasks:

- follow maintainer guidance from the GitHub Discussion
- use `docs/tt-metalium-qmul-design.md` as the implementation contract
- start with `[N, 4]` layout
- compare against CPU/PyTorch and scalar references
- report latency, throughput, numerical error, estimated FLOPs/sec, effective GB/sec, and arithmetic intensity

Exit criteria:

- minimal example or external prototype runs
- result is reproducible
- no unsupported claims about Tenstorrent performance

### Phase 3: TT-NN Wrapper

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

### Phase 4: TT-MLIR Lowering Discussion

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

### Phase 5: Hardware Report

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

1. Monitor `tt-awesome` issue #104 for maintainer labeling, approval, or entry
   PR generation.
2. Continue the `tenstorrent/tt-metal` GitHub Discussion if maintainers reply.
3. Post or refresh a short Discord note pointing to the Discussion only if it
   helps route maintainers to the narrow placement question.
4. If there is still no Discussion reply after the waiting period, open a
   shorter `tt-metal` issue:

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
