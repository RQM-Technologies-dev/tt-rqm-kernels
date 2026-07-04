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
- local tracker issue #8 now has its core technical evidence: a real
  experimental TT-Metalium `qmul` candidate built against `tt-metal/build_emule`
  and produced an emulation-labeled StructuredBench report
- x86-64 Linux preflight has passed inside Docker Desktop using sibling
  `tt-metal` and `tt-emule` checkouts:
  - `tt-metal` checkout: `/Users/home/Documents/tt-metal` at `fd810266`
  - `tt-emule` checkout: `/Users/home/Documents/tt-emule` at `abdc348`
  - Docker image: `python:3.12-slim` with `--platform linux/amd64`
- experimental TT-Metalium scalar RISC-V `qmul` candidate source has been
  added under `experimental/tt_metalium_qmul/`
- the new build-prerequisite gate is
  `experimental/tt_emule_qmul/check_build_prereqs.py`
- `tt-metal/build_emule` configure and install helpers now live under
  `experimental/tt_emule_qmul/`
- tt-emule candidate state:
  - `tt-emule` pins `tt-metal` commit
    `dd2849b5bc6b7a5d38a9eafbeba31ef8d530f8d4`
  - completed now: local `/Users/home/Documents/tt-metal` is detached at the
    pinned commit
  - completed now: required `tt-metal` submodules `umd` and `tracy` are
    initialized
  - completed now: `tt-metal/build_emule` configures successfully inside an
    Ubuntu 24.04 container with `TT_METAL_USE_EMULE=ON`
  - completed now: the build-prerequisite checker accepts both current
    lowercase `tt-metalium-config.cmake` package naming and installed-style
    `TT-MetaliumConfig.cmake` package naming
  - completed now: the experimental candidate builder expands uninstalled
    `build_emule` dependency prefixes so dependency lookup reaches `fmt`,
    `nlohmann_json`, `spdlog`, `tt-logger`, and UMD build outputs
  - completed now: `cmake --build build_emule --target install -j2` completed
    enough to install a usable TT-Metalium CMake package under
    `/Users/home/Documents/tt-metal/build_emule/lib/cmake/tt-metalium`
  - completed now: the build-prerequisite checker passes inside Ubuntu 24.04
    with the installed TT-Metalium package and required UMD/tracy submodules
  - completed now: the experimental candidate builder prefers the installed
    package directory over stale root-level generated config files
  - completed now: the candidate links against the current exported Metalium
    package plus explicit MPI, hwloc, and numa runtime dependencies needed by
    the installed `libtt_metal.so` / UMD exports
  - completed now: `experimental/tt_metalium_qmul/run_candidate_docker.sh`
    runs the built Linux candidate from the macOS StructuredBench harness
    through Docker and `tt-emule`
  - completed now: the candidate kernel uses the tt-emule-supported direct
    `reinterpret_cast<T*>(get_arg_val<uint32_t>(N))` L1 pointer pattern so the
    JIT can translate L1 scratch addresses
  - emulation report artifact:
    `reports/tt_emule_qmul_candidate.json` and
    `reports/tt_emule_qmul_candidate.md`
  - report label: `backend=external-qmul`,
    `device=tt-emule/tt-metalium-riscv-qmul-candidate`,
    `execution_label=emulation`, `stable_benchmark=false`
  - sample report scope: three 32-item `qmul` cases, `iters=1`, `warmup=0`;
    this is correctness/emulation evidence, not a stable performance benchmark
- local tracker issues #9 through #13 are closed after their design/demo
  deliverables landed
- local tracker issue #14 is open for the external LWT/ILWT `tt-metal`
  worktree path, which should stay outside this repo

The completed setup work should now be treated as background. The active work is
maintainer placement guidance and the first real lower-stack implementation path.

## Recommended Next Step

The x86-64 Linux Docker preflight is complete, `tt-metal/build_emule` has an
installed/exported TT-Metalium package, and the experimental scalar RISC-V
TT-Metalium `qmul` candidate now validates through `external-qmul` under
`tt-emule`.

The next step is to commit/push this evidence, update/close tracker issue #8
with the exact emulation report, then move the active implementation pressure to
issue #7: running the same external-qmul report path in a real Tenstorrent Cloud
or hardware environment. Placement issue #6 remains open in parallel.

Known-good preflight commands:

```bash
docker run --rm --platform linux/amd64 \
  -v /Users/home/Documents:/work \
  -w /work/tt-rqm-kernels \
  python:3.12-slim \
  python scripts/repo_status.py

docker run --rm --platform linux/amd64 \
  -v /Users/home/Documents:/work \
  -w /work/tt-rqm-kernels \
  -e TT_METAL_HOME=/work/tt-metal \
  -e TT_EMULE_HOME=/work/tt-emule \
  python:3.12-slim \
  python experimental/tt_emule_qmul/check_environment.py
```

Build-prerequisite command:

```bash
docker run --rm --platform linux/amd64 \
  -v /Users/home/Documents:/work \
  -w /work/tt-rqm-kernels \
  -e TT_METAL_HOME=/work/tt-metal \
  -e TT_EMULE_HOME=/work/tt-emule \
  python:3.12-slim \
  python experimental/tt_emule_qmul/check_build_prereqs.py
```

Configure command:

```bash
docker run --rm --platform linux/amd64 \
  -v /Users/home/Documents:/work \
  -w /work/tt-metal \
  -e TT_METAL_HOME=/work/tt-metal \
  -e TT_EMULE_HOME=/work/tt-emule \
  ubuntu:24.04 \
  bash /work/tt-rqm-kernels/experimental/tt_emule_qmul/configure_build_emule.sh
```

Build/install command:

```bash
docker run --rm --platform linux/amd64 \
  -v /Users/home/Documents:/work \
  -w /work/tt-metal \
  -e TT_METAL_HOME=/work/tt-metal \
  -e TT_EMULE_HOME=/work/tt-emule \
  -e JOBS=2 \
  ubuntu:24.04 \
  bash -lc 'bash /work/tt-rqm-kernels/experimental/tt_emule_qmul/configure_build_emule.sh && \
            bash /work/tt-rqm-kernels/experimental/tt_emule_qmul/build_install_emule.sh'
```

Candidate build command:

```bash
docker run --rm --platform linux/amd64 \
  -v /Users/home/Documents:/work \
  -w /work/tt-rqm-kernels \
  -e TT_METAL_HOME=/work/tt-metal \
  -e TT_EMULE_HOME=/work/tt-emule \
  ubuntu:24.04 \
  bash -lc 'apt-get update >/tmp/apt-update.log 2>&1 && \
            apt-get install -y --no-install-recommends ca-certificates cmake ninja-build g++ git python3 libopenmpi-dev openmpi-bin libhwloc-dev libnuma-dev >/tmp/apt-install.log 2>&1 && \
            python3 experimental/tt_metalium_qmul/build_candidate.py \
              --tt-metal-root /work/tt-metal \
              --cmake-prefix-path /work/tt-metal/build_emule \
              --build-dir /work/tt-rqm-kernels/experimental/tt_metalium_qmul/build_emule_candidate \
              --generator Ninja'
```

Emulation validation command:

```bash
python scripts/validate_qmul_candidate.py \
  --command "bash experimental/tt_metalium_qmul/run_candidate_docker.sh" \
  --execution-label emulation \
  --methodology-note "Experimental TT-Metalium qmul candidate run through tt-emule Docker wrapper; first validation sample, not a stable hardware benchmark." \
  --items 32 \
  --iters 1 \
  --warmup 0 \
  --json-output reports/tt_emule_qmul_candidate.json \
  --markdown-output reports/tt_emule_qmul_candidate.md
```

Preflight result:

```text
tt-metal root detected: /work/tt-metal
tt-emule root detected: /work/tt-emule
tt-emule qmul preflight passed. This does not run a kernel.
```

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
- the TT-Metalium candidate package now contains a real experimental scalar
  RISC-V source candidate that has built and run under tt-emule through
  `external-qmul`
- the Linux/tt-emule preflight now passes, `build_emule` has installed usable
  TT-Metalium exports, and the candidate has produced an emulation-labeled
  StructuredBench report through `external-qmul`
- the runbook now makes #7 actionable once Tenstorrent Cloud, a local
  TT-Metalium SDK checkout, or maintainer-provided environment guidance is
  available
- the repo status command and report metadata now make the current gap explicit:
  there is still no real Tenstorrent hardware report

## Priority Lanes

| Priority | Lane | Goal | Success condition |
| ---: | --- | --- | --- |
| 1 | TT-Metalium `qmul` placement | Choose the right lower-stack contribution path | Maintainers indicate whether the first candidate should live externally, as a TT-Metalium example, or another preferred route |
| 2 | tt-emule environment and candidate validation | Prove the candidate can build/run without hardware first | Complete: `build_emule` installs a TT-Metalium package and the candidate emits an emulation-labeled report |
| 3 | TT-Metalium `qmul` example | Prove RQM can operate at the lower stack | Experimental `[N, 4]` `qmul` candidate compared against CPU/PyTorch and scalar references through `external-qmul`; upstream placement still open |
| 4 | StructuredBench report standard | Make this useful as a reusable benchmark class | CPU, TT-Lang, emulation, and future hardware reports share `structuredbench.v1` fields with explicit execution labels |
| 5 | Cloud/hardware validation | Turn the benchmark into performance evidence | First Tenstorrent hardware report compares CPU/PyTorch vs Tenstorrent backend |
| 6 | TT-NN wrapper | Make kernels usable by ordinary Tenstorrent developers | `qmul` or `qrotate_vector` exposed through a TT-NN-style wrapper after lower-stack proof |
| 7 | TT-MLIR lowering discussion | Explore compiler value after a working kernel exists | Concrete question: should `qmul` lower as a fused kernel instead of scalar expansion? |

## Next Repo Work

### 1. Track TT-Metalium Placement

Active repo issues should now focus on the hardware-facing path:

1. `Track TT-Metalium qmul placement guidance` (#6)
2. `Implement minimal TT-Metalium qmul example using the external-qmul harness` (#3)
3. `Run StructuredBench on Tenstorrent Cloud` (#7)
4. `Define TT-NN wrapper path after lower-stack qmul proof` (#4)
5. `Set up external tt-metal LWT/ILWT worktree path` (#14, separate from this
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

- follow maintainer guidance from the GitHub Discussion when it arrives
- use `docs/tt-metalium-qmul-design.md` as the implementation contract
- use the `external-qmul` harness as the validation bridge for a standalone
  candidate executable
- use `experimental/tt_metalium_qmul/` as the external staging location until a
  maintainer requests another placement
- completed: use `docs/tt-emule-qmul-validation-plan.md`,
  `experimental/tt_emule_qmul/check_environment.py`, and the Docker wrapper
  before claiming emulation readiness
- start with `[N, 4]` layout
- completed for tt-emule: compare against CPU/PyTorch and scalar references
- completed for tt-emule: report latency, throughput, numerical error, estimated
  FLOPs/sec, effective
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
