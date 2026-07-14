# SU2ComposeBench eligibility promotion audit

Audit date: `2026-07-14`

Promoted source: `3238299a9eea2a44dccd6826a947cac3266dd2f7`

Rebuilt candidate SHA-256: `d8237f2e5b05885167085d87a0400daf8b5feb0318d906285a1d263035294441`

Pinned TT-Metal: `dd2849b5bc6b7a5d38a9eafbeba31ef8d530f8d4`

## Decision

The candidate is `performance_eligible=true` after the separately committed
pre-eligibility conformance report and architecture audit. The rebuilt binary
then passed the same B=32/K=8 and B=2048/K=8 whole-output gates with zero
failing and zero nonfinite values on N300 device 0.

Eligibility describes the audited implementation architecture. It does not
set `stable_benchmark=true` and does not establish an acceleration claim.

## Reconfirmed boundaries

- One Wormhole device: device 0 only; one create and one close.
- Row-major multicore split selected two Tensix cores for B=2048.
- Quaternion and phase arithmetic is confined to compute/SFPU kernels.
- Reader and writer kernels perform DMA and synchronization only.
- The fused accumulator remains in L1 ping-pong storage without intermediate
  DRAM writes.
- Fused dispatch count is one; unfused dispatch count is K-1.
- Candidate provenance names the rebuilt binary hash above, not a command
  wrapper or environment executable.
- The performance report remains a one-session Claim Level 1 sample with
  `stable_benchmark=false`.
