# TT-MLIR Fused Lowering Prerequisites

## Purpose

This document records when a future TT-MLIR fused lowering RFC would be
appropriate for `tt-rqm-kernels`.

It is not an RFC. It is a guardrail to avoid opening a compiler-facing proposal
before there is enough backend evidence.

## Future Question

The future compiler question is:

```text
Should qmul lower as a fused structured operator rather than scalar expansion?
```

That question is only useful after `qmul` has evidence below CPU/PyTorch.

## Evidence Required Before An RFC

Before writing or posting a TT-MLIR RFC, the repo should have at least one of:

- a real TT-Metalium `qmul` candidate report
- a tt-emule run of a real TT-Metalium `qmul` candidate
- maintainer guidance that TT-MLIR is the preferred next discussion path
- Tenstorrent hardware or cloud results using the external `qmul` harness

TT-Lang simulator results are useful lower-stack evidence, but they should not
be treated as hardware performance or as proof that TT-MLIR should add a fused
operation.

## Current Status

The repo now has a tt-emule run of the experimental TT-Metalium `qmul`
candidate, labeled as emulation and not hardware performance. That satisfies
one evidence prerequisite for future compiler discussion, but it is still not a
reason to open a TT-MLIR proposal by itself. The next useful inputs are a real
hardware run, a concrete comparison against scalar-expanded lowering, or
actionable maintainer guidance if it arrives.

## Minimum RFC Inputs

A future RFC should include:

- the `[N, 4] x [N, 4] -> [N, 4]` `qmul` operator contract
- Hamilton product lane equations
- CPU/PyTorch reference error results
- scalar reference spot-check results
- TT-Lang simulator notes labeled simulator-only
- tt-emule, TT-Metalium, or hardware report fields if available
- comparison against scalar-expanded lowering
- a clear non-goal that this is not a native quaternion datatype request

## Non-Goals

- No TT-MLIR proposal in this milestone.
- No compiler-facing issue before backend evidence exists.
- No native quaternion datatype request.
- No new Tenstorrent hardware feature request.
- No claim that simulator or CPU results predict hardware performance.
- No Tenstorrent endorsement claim.
