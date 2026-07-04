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
- StructuredBench reports now include explicit `execution_label`,
  `stable_benchmark`, and `methodology_note` fields so CPU, simulator,
  emulation, and hardware samples are labeled directly
- `python scripts/repo_status.py` gives a one-command current status summary
- optional TT-Lang simulator `qmul` prototype with a hardened,
  simulator-only StructuredBench-compatible report
- an `external-qmul` candidate harness for validating future standalone
  `qmul` executables against CPU/PyTorch and scalar references
- an external TT-Metalium candidate staging package under
  `experimental/tt_metalium_qmul/` with build/run/validation placeholders
- Tenstorrent-facing docs, operator contracts, outreach packet, and CI
- a Tenstorrent execution runbook and hardware report template for future
  StructuredBench `qmul` runs
- a minimal TT-Metalium `qmul` design document for `[N, 4]` structured tensors
- scientific/HPC positioning that relates RQM structured kernels to
  Tenstorrent's broader non-LLM scientific workload direction
- a developer tutorial for structured `[N, 4]` `qmul` kernels:
  `docs/structured-qmul-tutorial.md`
- a GitHub Discussion opened in `tenstorrent/tt-metal`
- a narrower `tt-metal` placement issue opened after the Discussion waiting
  period:
  https://github.com/tenstorrent/tt-metal/issues/48944
- `tt-awesome` submission issue #104 approved, with generated entry PR #106
  merged:
  https://github.com/tenstorrent/tt-awesome/pull/106
- local tracker issue #1 closed after the generated entry PR merged
- local tracker issue #2 closed after hardening the TT-Lang simulator report
- local tracker issue #3 started with an external TT-Metalium candidate scaffold
- local tracker issue #7 started with execution runbook/report-template prework
- local tracker issue #8 is open for tt-emule validation of a real
  TT-Metalium `qmul` candidate
- local tracker issues #9 through #13 are closed after their design/demo
  deliverables landed
- local tracker issue #14 is open for the external LWT/ILWT `tt-metal`
  worktree path, which should stay outside this repo

The completed setup work should now be treated as background. The active work is
maintainer placement guidance and the first real lower-stack implementation path.

## Recommended Next Step

Provision an x86-64 Linux environment with `tt-metal`, `tt-emule`, and
`tt-rqm-kernels` checked out together, then run:

```bash
python scripts/repo_status.py
python experimental/tt_emule_qmul/check_environment.py
```

Record the exact blocker or success in issue #8. After that, build the smallest
possible real TT-Metalium `qmul` host/kernel candidate and validate it through
`external-qmul`.

Why this is next:

- the repo now has CPU/PyTorch reference results, a TT-Lang simulator proof
  point, and a public collaboration map
- the `tt-awesome` submission has merged, so it is no longer an active
  repo-building blocker
- the minimal TT-Metalium `qmul` design document is present, so the next step is
  coordination and implementation planning rather than more design prose
- the scientific/HPC positioning now gives RQM a conservative way to reference
  adjacent Tenstorrent scientific workload activity without claiming that
  spectral element methods need quaternion kernels
- the external `qmul` harness gives future TT-Metalium, TT-NN, or cloud-hosted
  candidate executables a concrete way to emit comparable StructuredBench
  reports
- the TT-Metalium candidate package is staged externally and avoids unverified
  TT-Metalium source until a real SDK/hardware environment is available
- the candidate scaffold now has explicit build/run/validation commands, so the
  remaining #3 work is real TT-Metalium host/kernel implementation
- the runbook now makes #7 actionable once Tenstorrent Cloud, a local
  TT-Metalium SDK checkout, or maintainer-provided environment guidance is
  available
- the repo status command and report metadata now make the current gap explicit:
  there is no real TT-Metalium `qmul` candidate yet

## Priority Lanes

| Priority | Lane | Goal | Success condition |
| ---: | --- | --- | --- |
| 1 | TT-Metalium `qmul` placement | Choose the right lower-stack contribution path | Maintainers indicate whether the first candidate should live externally, as a TT-Metalium example, or another preferred route |
| 2 | tt-emule environment and candidate validation | Prove the candidate can build/run without hardware first | x86-64 Linux preflight passes and a real TT-Metalium candidate emits an emulation-labeled report |
| 3 | TT-Metalium `qmul` example | Prove RQM can operate at the lower stack | Minimal `[N, 4]` `qmul` kernel compared against CPU/PyTorch and scalar references through `external-qmul` |
| 4 | StructuredBench report standard | Make this useful as a reusable benchmark class | CPU, TT-Lang, emulation, and future hardware reports share `structuredbench.v1` fields with explicit execution labels |
| 5 | Cloud/hardware validation | Turn the benchmark into performance evidence | First Tenstorrent hardware report compares CPU/PyTorch vs Tenstorrent backend |
| 6 | TT-NN wrapper | Make kernels usable by ordinary Tenstorrent developers | `qmul` or `qrotate_vector` exposed through a TT-NN-style wrapper after lower-stack proof |
| 7 | TT-MLIR lowering discussion | Explore compiler value after a working kernel exists | Concrete question: should `qmul` lower as a fused kernel instead of scalar expansion? |

## Next Repo Work

### 1. Track TT-Metalium Placement

Active repo issues should now focus on the hardware-facing path:

1. `Track TT-Metalium qmul placement guidance` (#6)
2. `Implement minimal TT-Metalium qmul example using the external-qmul harness` (#3)
3. `Add tt-emule validation milestone for qmul` (#8)
4. `Run StructuredBench on Tenstorrent Cloud` (#7)
5. `Define TT-NN wrapper path after lower-stack qmul proof` (#4)
6. `Set up external tt-metal LWT/ILWT worktree path` (#14, separate from this
   repo's implementation track)

Each issue should include:

- goal
- acceptance criteria
- references to relevant docs
- clear non-goals

## Technical Roadmap

### Phase 1: Ecosystem Visibility

Goal:

- make `tt-rqm-kernels` discoverable inside the Tenstorrent ecosystem

Status:

- complete: `tt-awesome` submission issue #104 was approved and generated PR
  #106 merged

Exit criteria:

- generated `tt-awesome` entry PR #106 merged
- public description leads with structured-kernel benchmarking, not quaternion theory

### Phase 2: TT-Metalium `qmul`

Goal:

- create the first real lower-stack kernel target

Tasks:

- follow maintainer guidance from the GitHub Discussion
- use `docs/tt-metalium-qmul-design.md` as the implementation contract
- use the `external-qmul` harness as the validation bridge for a standalone
  candidate executable
- use `experimental/tt_metalium_qmul/` as the external staging location until a
  maintainer requests another placement
- use `docs/tt-emule-qmul-validation-plan.md` and
  `experimental/tt_emule_qmul/check_environment.py` before claiming emulation
  readiness
- start with `[N, 4]` layout
- compare against CPU/PyTorch and scalar references
- report latency, throughput, numerical error, estimated FLOPs/sec, effective
  GB/sec, arithmetic intensity, `execution_label`, `stable_benchmark`, and
  `methodology_note`

Exit criteria:

- minimal example or external prototype runs
- result is reproducible
- emulation reports are labeled `execution_label=emulation`
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

Current outreach state:

- Discussion #48871 is open in `tenstorrent/tt-metal` with no maintainer
  comments as of July 3, 2026:
  https://github.com/tenstorrent/tt-metal/discussions/48871
- Narrow placement issue #48944 is open in `tenstorrent/tt-metal`:
  https://github.com/tenstorrent/tt-metal/issues/48944

Next outreach actions:

1. Continue the Discussion or issue if Tenstorrent maintainers reply.
2. Post or refresh a short Discord note pointing to the narrow placement issue
   only if it helps route maintainers to the question.
3. Contact Tenstorrent OSPO only after the repo has a technical packet plus
   either a TT-Lang simulation or a clear maintainer response.

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
