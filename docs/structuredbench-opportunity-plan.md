# StructuredBench Opportunity Plan

## Summary

`tt-rqm-kernels` should treat `qmul` as the wedge and StructuredBench as the
main asset.

The repo is no longer only a quaternion reference demo. It is becoming RQM
Technologies' entry point into structured scientific, geometric, signal, and
simulation kernels on open AI accelerators. The public Tenstorrent ask should
remain narrow for now:

```text
Where should a minimal [N, 4] TT-Metalium qmul example live?
```

The broader repo direction is:

```text
CPU/PyTorch reference
-> scalar correctness checks
-> TT-Lang simulator
-> tt-emule-compatible TT-Metalium qmul
-> Tenstorrent Cloud / hardware report
-> TT-NN and TT-MLIR discussions after backend evidence
```

This keeps the project useful to Tenstorrent without asking for native
quaternion hardware, new silicon features, or endorsement.

## Current Position

The repo already has the right handshake:

- CPU/PyTorch quaternion, rotor, and phase reference kernels
- scalar spot checks for independent correctness
- StructuredBench reports with latency, throughput, numerical error, estimated
  FLOPs/sec, effective GB/sec, and arithmetic intensity
- TT-Lang simulator `qmul`, including simulator-only trace/stat support
- external `qmul` harness for future candidate executables
- TT-Metalium staging package without fake TT-Metalium source
- tt-emule qmul validation plan and local preflight scaffold
- tt-emule tracker issue #8:
  <https://github.com/RQM-Technologies-dev/tt-rqm-kernels/issues/8>
- ComplexTensor-to-QuaternionTensor bridge design and tracker issue #9:
  <https://github.com/RQM-Technologies-dev/tt-rqm-kernels/issues/9>
- `phase_update` Tenstorrent backend plan and tracker issue #10:
  <https://github.com/RQM-Technologies-dev/tt-rqm-kernels/issues/10>
- selected external Tenstorrent contribution path and tracker issue #11:
  <https://github.com/RQM-Technologies-dev/tt-rqm-kernels/issues/11>
- public `tt-metal` placement Discussion and narrow placement issue
- `tt-awesome` ecosystem visibility

The missing opportunity is to make StructuredBench the reusable benchmark
family:

```text
structured 4-lane values
quaternion multiply
rotor/vector rotation
normalization and inverse stability
phase/orientation update
small wave/stencil-style updates
complex/quaternion bridge patterns
physical-AI pose streams
```

## Priority Roadmap

### 1. Physical-AI Pose Stream Demo

Goal: give `qrotate_vector` a practical robotics/sensing story.

Build a small demo path:

```text
model/vector stream
-> qrotate_vector / rotor update
-> structured physical state
-> StructuredBench-style report
```

This should remain a benchmark/demo, not a full robotics application.

Exit condition:

- a reproducible example that shows why rotor/vector kernels matter for
  physical AI and pose/orientation streams

### 2. StructuredBench-HPC Expansion

Goal: broaden the benchmark family beyond quaternions while preserving the repo
identity.

Candidate future workloads:

- small wave/stencil update kernels
- compact N-body-like vector state updates
- complex/quaternion bridge kernels
- phase/magnitude update streams
- orientation/pose-update streams

Exit condition:

- a staged roadmap that adds one workload at a time with CPU/PyTorch reference,
  scalar or independent checks where possible, and backend-comparable reports

### 3. TT-MLIR Fused Lowering RFC

Goal: ask the compiler question only after backend evidence exists.

The future question:

```text
Should qmul lower as a fused structured operator rather than scalar expansion?
```

Do not lead with this. Prepare the story now, but only open a compiler-facing
discussion after there is tt-emule, TT-Metalium, or hardware evidence.

Exit condition:

- a short RFC grounded in measured backend behavior, not speculation

## Recommended Issues To Add

Add these tracker issues when ready:

1. `Draft physical-AI pose stream demo using qrotate_vector`
2. `Draft StructuredBench-HPC expansion roadmap`

The first issue should be the next concrete technical move.

## Non-Goals

- Do not broaden the public Tenstorrent ask before placement issue feedback.
- Do not ask for native quaternion hardware.
- Do not ask for new chip features.
- Do not imply Tenstorrent endorsement.
- Do not claim CPU, simulator, or emulation results as hardware performance.
- Do not lead with defense; keep it as a downstream application area.
- Do not open TT-MLIR or TT-NN asks before lower-stack evidence exists.

## References

- Tenstorrent software stack:
  <https://docs.tenstorrent.com/getting-started/tt-software-stack.html>
- TT-Metalium documentation:
  <https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/>
- tt-emule:
  <https://github.com/tenstorrent/tt-emule>
- tt-awesome:
  <https://github.com/tenstorrent/tt-awesome>
- TT-MLIR:
  <https://github.com/tenstorrent/tt-mlir>
