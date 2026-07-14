# Stage B qmul Internal Engineering Review

Date: 2026-07-14  
Reviewer: RQM Technologies internal engineering review  
Scope: `main...codex/stage-b-qmul` for draft PR #16

This is an RQM internal engineering review. It is not a Tenstorrent review,
endorsement, stability determination, or acceleration claim.

## Files inspected

- `experimental/tt_metalium_qmul/src/qmul_multicore_candidate.cpp`
- `experimental/tt_metalium_qmul/kernels/qmul_multicore_reader.cpp`
- `experimental/tt_metalium_qmul/kernels/qmul_multicore_compute.cpp`
- `experimental/tt_metalium_qmul/kernels/qmul_multicore_writer.cpp`
- `experimental/tt_metalium_qmul/kernels/qmul_sfpu.h`
- `experimental/tt_metalium_qmul/CMakeLists.txt`
- `experimental/tt_metalium_qmul/build_candidate.py`
- `experimental/tt_metalium_qmul/validate_candidate.py`
- `scripts/validate_qmul_candidate.py`
- `tt_rqm_kernels/benchmark_integrity.py`
- `tt_rqm_kernels/structuredbench.py`
- the Stage B tests, reports, architecture audit, runbook, design document,
  landing page, and README changes
- pinned TT-Metalium/SFPI sources at
  `dd2849b5bc6b7a5d38a9eafbeba31ef8d530f8d4`, especially the Wormhole binary
  and ternary SFPU parameter wrappers, `llk_math_eltwise_sfpu_common.h`,
  `work_split.cpp`, and `core_coord.cpp`

## Stage A immutability

The Stage A scalar host source, scalar RISC-V kernel, JSON report, Markdown
report, and environment record have identical Git object IDs on `main` and the
reviewed branch. The scalar executable target name and default build selection
also remain intact. Stage B is a separate CMake executable and must be selected
explicitly with `--candidate multicore`.

## Arithmetic path

The reader performs page addressing, CB reservation, asynchronous reads, one
read barrier per tile, and CB publication. The writer performs CB waits, page
addressing, asynchronous writes, one write barrier per tile, and CB retirement.
Neither data-movement kernel includes SFPI or Hamilton-product arithmetic.

The compute kernel copies operand tiles into destination registers and invokes
the product, add-product, and subtract-product SFPU helpers. The four component
sign patterns match the Hamilton product. `ComputeConfig` enables
`fp32_dest_acc_en`, the eight input CBs use `UnpackToDestFp32`, and all input and
output CBs use `DataFormat::Float32` with a 4,096-byte page for one 32x32 tile.

## SFPU tile-coverage proof

For the pinned Wormhole implementation:

1. `_llk_math_eltwise_sfpu_start_(0)` establishes the destination base.
2. `VectorMode::RC` calls the supplied SFPU helper four times, once for each
   tile face.
3. Between calls, `_llk_math_eltwise_sfpu_inc_dst_face_addr_()` performs two
   `math::inc_dst_addr<8>()` operations, advancing to the next face.
4. Each qmul helper call visits `i=0..7`, so it processes eight destination
   vectors in that face.
5. One Wormhole SFPI `vFloat` operation covers 32 lanes. A face therefore
   covers `8 * 32 = 256` elements.
6. The four face bases are non-overlapping, so the union is 32 vectors and
   `4 * 8 * 32 = 1,024` unique elements: exactly one complete 32x32 tile.
7. Tile operands are addressed in 32-vector strides. Inputs 0, 1, and 2
   therefore remain distinct while the face base advances in lockstep.

The product and each signed accumulation use the same RC wrapper, so this proof
applies independently to every Hamilton-product term and all four output
components. Focused structural tests extract the helper constants, require all
three wrappers to use `VectorMode::RC`, and enumerate the 1,024 unique
vector/lane positions. No partial-face defect was found.

## Layout and boundary cases

The host calculates `component_tiles = ceil(N / 1024)`, allocates four padded
component planes per operand, initializes padding to zero, and maps AoS element
`[item, lane]` to planar `[lane, item]`. Readback applies the inverse mapping
only for `item < N`, discarding padding.

The formulas cover the smallest legal `N=1`, `N=128`, exact tile multiples,
non-multiples such as 1,025 and 65,537, and all official Stage B sizes. A
focused round-trip model now exercises those cases and checks that every padded
cell remains zero.

## Work split and runtime arguments

`split_work_to_cores(grid, component_tiles, true)` selects
`min(component_tiles, grid.x * grid.y)` cores. At the pinned commit, `true`
constructs the groups row-wise, and `CoreRange` iteration advances x before y.
The host walks the higher-work group followed by the lower-work group, assigns
each core a contiguous `[start_tile, start_tile + tiles_per_core)` interval,
and advances `start_tile` by exactly that interval length. The final equality
check against `component_tiles` prevents omission. The split groups are
disjoint by construction, preventing overlap. Reader, compute, and writer
receive the same per-core count; reader and writer also receive the same start
tile and component-plane stride.

The committed reports observe one selected core at N=128 and 4, 56, and 56
selected cores for the three Stage B sizes on the measured 8x7 grid.

## Circular buffers and synchronization

Each of the twelve CBs has two one-tile pages. The reader reserves all eight
input pages before issuing reads and publishes them only after the read
barrier. Compute waits on all eight inputs, produces and publishes each of four
outputs, and pops the inputs only after all four components are complete. The
writer waits on all four outputs, retires them only after its write barrier,
and therefore cannot expose an output page for reuse while DMA is live.

The two-page capacity permits bounded overlap without overwriting live data.
Backpressure terminates at the writer and every reserve/wait has a downstream
push/pop counterpart. No deadlock or live-page overwrite path was found.

## Timing scope and workload reuse

Input conversion, unit-mesh creation, DRAM allocation, host-to-device writes,
program construction, JIT preparation, and an initial `Finish` are included in
`setup_s`. Warmups use blocking workload enqueue and occur before the device
timer. The measured loop reuses the prepared `MeshWorkload`; each of 30
iterations is enqueued with blocking enabled, followed by `Finish`. Thus
`device_s` includes host enqueue/dispatch synchronization plus device execution
for the prepared workload, but excludes setup, readback, device close, and
process startup. The report's timer-scope string says this. End-to-end timing is
measured independently around the subprocess and includes all of those costs.

This first method is internally consistent but not yet a steady-state service
measurement. Persistent-device timing is deferred to the stabilization issue.

## Metrics and provenance

The candidate obtains grid dimensions and selected-core count from the created
unit mesh and work split; the validator independently recomputes component
tiles, available cores, and selected cores. Source inspection proves the
reported one-device restriction, planar layout, and compute/SFPU arithmetic
path rather than accepting those strings alone.

The conformance report identifies execution-source commit `7a9b6b0`, candidate
binary SHA-256 `16ab3861...`, and `performance_eligible=false`. The performance
report identifies promotion commit `debdce2`, candidate binary SHA-256
`af48c791...`, and `performance_eligible=true`. Both record the pinned
TT-Metalium commit and exact environment fields, and both retain
`stable_benchmark=false`.

TT-Metalium JIT-compiles the device kernel sources at workload construction,
so the host-executable hash is not by itself a hash of JIT output. The evidence
is therefore bound jointly by the exact host hash, clean execution-source
commit, pinned TT-Metalium commit, and environment record. The reviewed source
trees were clean. A future persistent measurement schema may add an explicit
kernel-bundle or JIT-artifact digest without changing the meaning of the
existing `candidate_sha256` field.

## Defects found and repaired

The review found policy and wrapper defects, not a device-arithmetic defect:

- the convenience validator did not expose or propagate `--candidate`, making
  correct protected Stage B selection impossible through that wrapper;
- an explicit conformance stage could be labeled as emulation or CPU;
- candidate selection was not checked against the implementation class emitted
  by the hardware report;
- conformance did not enforce `performance_eligible=false`; and
- performance did not independently require the multicore/SFPU implementation
  class after checking eligibility.

These gates and focused regressions were added. No CMake target, C++ host code,
device kernel, arithmetic path, timing implementation, work split, candidate
binary, or existing report was changed. The Stage B hardware evidence therefore
does not require regeneration; it is revalidated under the stricter policy.

## Remaining uncertainties

- Tenstorrent has not reviewed whether the SFPU destination-register usage,
  CB choices, work split, or timer scope is idiomatic for Wormhole.
- The current subprocess-per-repetition method has substantial cold setup and
  device lifecycle overhead; a persistent-device path is still required.
- One hardware session is insufficient for `stable_benchmark=true`.
- No controlled, timing-scope-compatible CPU comparison exists.
- Device 1 was not used and no dual-device conclusion is supported.

## Merge recommendation

The implementation is technically suitable to merge as experimental,
one-device Stage B work after the updated local suite, report-integrity checks,
remote health/cleanliness checks, and PR CI all pass. Merge does not authorize
`stable_benchmark=true`, an acceleration claim, CPU/N300 comparison, official
Tenstorrent endorsement, or use of the second N300 device.
