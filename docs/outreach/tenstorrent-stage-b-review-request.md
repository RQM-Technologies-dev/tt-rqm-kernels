# Draft: Tenstorrent Stage B qmul Engineering Review Request

Status: draft; do not send without an explicit repository-supported outreach
step. This document does not imply Tenstorrent review or endorsement.

## Context

RQM Technologies implemented an experimental Float32 Hamilton-product kernel
for one Wormhole device in `tt-rqm-kernels` PR #16. The candidate converts
`[N,4]` AoS inputs to padded planar 32x32 tiles, distributes component tiles
row-major across Tensix cores, performs arithmetic in a compute/SFPU kernel,
and uses data-movement RISC-V kernels only for DMA and synchronization.

The N=128 conformance gate and one official three-size sample passed on N300
device 0 against TT-Metalium commit
`dd2849b5bc6b7a5d38a9eafbeba31ef8d530f8d4`. The first sample deliberately
remains `stable_benchmark=false`. No acceleration or dual-device claim is being
made.

## Narrow review request

Could a Tenstorrent engineer review whether the following choices are correct
and idiomatic for Wormhole?

1. SFPU destination-register addressing: tile operands use 32-vector strides,
   each face helper visits eight vectors, and `VectorMode::RC` advances through
   all four faces.
2. Full tile coverage: four faces times eight vectors times 32 lanes covers one
   32x32 Float32 tile exactly once for each product/add/subtract term.
3. Float32 flow: Float32 CB pages, `UnpackToDestFp32`, FP32 destination
   accumulation, and Float32 packing back to the output CB.
4. Layout: four padded component planes per operand, each represented as
   contiguous 32x32 tiles.
5. Multicore split: `split_work_to_cores(..., true)` plus contiguous per-core
   component-tile intervals and group-specific runtime counts.
6. Buffering: two pages for each of eight input and four output CBs, with the
   current reserve/wait/push/pop and DMA barrier order.
7. Device scope: exactly one unit mesh on logical device 0; device 1 is outside
   the current work.
8. Timing: whether prepared workload enqueue plus synchronization is an
   appropriate first device scope, and what TT-Metalium mechanism is preferred
   for a persistent-device steady-state measurement.

## Review pointers

- `experimental/tt_metalium_qmul/src/qmul_multicore_candidate.cpp`
- `experimental/tt_metalium_qmul/kernels/qmul_multicore_reader.cpp`
- `experimental/tt_metalium_qmul/kernels/qmul_multicore_compute.cpp`
- `experimental/tt_metalium_qmul/kernels/qmul_multicore_writer.cpp`
- `experimental/tt_metalium_qmul/kernels/qmul_sfpu.h`
- `reports/tt_hardware_qmul_stage_b_internal_review.md`
- `reports/tt_hardware_qmul_stage_b_candidate_conformance.md`
- `reports/tt_hardware_qmul_stage_b_architecture_audit.md`
- `reports/tt_hardware_qmul_stage_b_performance.md`

## Questions we would especially value answers to

- Is direct `sfpi::dst_reg[tile_index * 32 + vector]` access under the pinned
  binary/ternary RC wrappers the preferred way to address multiple FP32 tiles?
- Does the RC face increment interact with FP32 destination mode exactly as the
  internal coverage proof assumes?
- Are two-page per-component CBs reasonable, or would fewer CBs, larger CBs, or
  another operand-staging pattern be more idiomatic?
- Is a reusable `MeshWorkload` with a persistent `MeshDevice`, events, traces,
  or device profiler counters the recommended steady-state timing path?
- Are there Wormhole-specific dispatch, pack/unpack, or synchronization details
  that the current correctness results could conceal?

The goal is to improve correctness confidence and measurement methodology
before any stability or comparative performance statement is considered.
