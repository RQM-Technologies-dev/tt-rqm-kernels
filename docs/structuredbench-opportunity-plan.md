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
- experimental TT-Metalium scalar RISC-V `qmul` candidate staged externally by
  default
- tt-emule qmul validation plan, local preflight scaffold, Docker/Linux
  source-tree preflight success, and emulation-labeled StructuredBench report
- tt-emule tracker issue #8:
  <https://github.com/RQM-Technologies-dev/tt-rqm-kernels/issues/8>
- ComplexTensor-to-QuaternionTensor bridge design and closed tracker issue #9:
  <https://github.com/RQM-Technologies-dev/tt-rqm-kernels/issues/9>
- `phase_update` Tenstorrent backend plan and closed tracker issue #10:
  <https://github.com/RQM-Technologies-dev/tt-rqm-kernels/issues/10>
- selected external Tenstorrent contribution path and closed tracker issue #11:
  <https://github.com/RQM-Technologies-dev/tt-rqm-kernels/issues/11>
- physical-AI pose stream demo and closed tracker issue #12:
  <https://github.com/RQM-Technologies-dev/tt-rqm-kernels/issues/12>
- StructuredBench-HPC expansion roadmap and closed tracker issue #13:
  <https://github.com/RQM-Technologies-dev/tt-rqm-kernels/issues/13>
- external LWT/ILWT `tt-metal` worktree tracker issue #14:
  <https://github.com/RQM-Technologies-dev/tt-rqm-kernels/issues/14>
- TT-MLIR fused lowering prerequisites documented:
  `docs/tt-mlir-fused-lowering-prerequisites.md`
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

No active opportunity-plan prose items remain.

The active work is implementation pressure beyond emulation: keep the
experimental TT-Metalium `qmul` candidate externally staged, keep its tt-emule
validation reproducible, and run the same `external-qmul` report path in a real
Tenstorrent Cloud or hardware environment.

The TT-MLIR fused lowering question remains deferred by design. The repo now has
tt-emule evidence, but the compiler question should still wait for a hardware
run, a concrete scalar-expanded comparison, or actionable maintainer guidance if
it happens to arrive.

## Non-Goals

- Do not broaden the public Tenstorrent ask while hardware validation is still
  missing.
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
