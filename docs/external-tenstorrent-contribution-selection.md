# External Tenstorrent Contribution Selection

## Purpose

This document records the first external Tenstorrent contribution path selected
for RQM Technologies while keeping `tt-rqm-kernels` focused on StructuredBench
and `qmul` placement.

Tracker issue:
<https://github.com/RQM-Technologies-dev/tt-rqm-kernels/issues/11>

Follow-on external worktree tracker:
<https://github.com/RQM-Technologies-dev/tt-rqm-kernels/issues/14>

Reviewed upstream context on July 4, 2026.

## Selected Path

Selected upstream issue:

- `tenstorrent/tt-metal#40494`
- `[Bounty $2.5k] Implement Lifting Wavelet Transform (LWT) and Inverse LWT (ILWT)`
- <https://github.com/tenstorrent/tt-metal/issues/40494>

This was selected because it is an operator-level scientific, signal,
imaging, and wave-processing workload. It asks for LWT/ILWT support as TT-NN
operations with optimized kernels, explicit shape/layout requirements, and
precision reporting against PyWavelets.

The selection is based on technical fit, not bounty amount.

## Why This Fits RQM

LWT/ILWT is adjacent to the same lower-stack themes that make StructuredBench
useful:

- structured numerical kernels below full applications
- signal and imaging workloads
- wave and multi-resolution state
- explicit correctness and numerical error reporting
- data movement and neighbor-access patterns
- future TT-NN or lower-stack operator credibility

This does not imply that wavelet transforms need quaternion kernels. The fit is
about shared engineering concerns: structured data movement, compact numerical
operators, and reproducible validation.

## Why Other Current Options Were Not First

Several current Tenstorrent bounties are model bring-up tasks such as audio,
voice, vision, or time-series model ports. Those are useful, but they are
application-level and less directly aligned with RQM's current lower-stack
kernel positioning.

Compiler or packaging work can also build credibility, but LWT/ILWT is a
closer technical match because it is a scientific/signal operator with explicit
kernel and error-reporting requirements.

## Execution Guidance

If RQM pursues this external contribution:

1. Create a separate `tt-metal` fork/worktree for upstream work.
2. Treat `tenstorrent/tt-metal#40494` as the source of truth.
3. Start by reproducing the requirement surface and reference validation against
   PyWavelets.
4. Scope a small first contribution or maintainer question before commenting
   upstream.
5. Keep all LWT/ILWT implementation work outside `tt-rqm-kernels`.

Good first maintainer question, once prepared:

```text
Would a narrow first slice for Haar/db1 1D row-major LWT with PyWavelets
validation be useful before attempting the broader wavelet catalog?
```

Do not post that question until the local scope is concrete enough to be useful.

## Relationship To StructuredBench

`tt-rqm-kernels` should continue pursuing:

- `qmul` placement guidance
- TT-Lang and tt-emule validation paths
- external candidate harnesses
- later `qrotate_vector`, `phase_update`, and pose-stream benchmarks

The LWT/ILWT path is an external credibility lane. It should not broaden the
public ask for `tt-rqm-kernels` or dilute the current `[N, 4]` structured
kernel story.

## Non-Goals

- No upstream PR or issue comment in this milestone.
- No claim that LWT/ILWT requires quaternion or rotor kernels.
- No LWT/ILWT implementation inside `tt-rqm-kernels`.
- No bounty-first framing.
- No Tenstorrent endorsement claim.
- No defense-first framing.
