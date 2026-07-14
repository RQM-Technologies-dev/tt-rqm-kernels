# Stage B qmul Architecture and Eligibility Audit

Date: 2026-07-14

This audit authorizes the first performance-eligible build of the multicore
TT-Metalium qmul candidate. It is an architecture classification, not a
benchmark-stability or acceleration claim.

## Conformance evidence

- Execution-source commit: `7a9b6b044e623e62938127136f61ebec54f06eb7`
- Conformance-evidence commit: `6e3e33f`
- Candidate SHA-256: `16ab386118bcf934ca01d1269ea69f3e1272be3de742f2f59ca7ec30d9a4cd8d`
- TT-Metalium commit: `dd2849b5bc6b7a5d38a9eafbeba31ef8d530f8d4`
- Hardware: N300, Wormhole device 0
- Whole-output result: 512 validated values, zero non-finite values, zero
  failing values, and maximum absolute error `9.686581936563243e-08` at
  `atol=1e-4`, `rtol=1e-4`
- Conformance state: `performance_eligible=false`, `stable_benchmark=false`

The report candidate hash exactly matched `sha256sum` of the executed binary.

## Kernel-role audit

- `qmul_multicore_reader.cpp` contains address calculation, CB coordination,
  `noc_async_read_page`, and read barriers only. It contains no SFPU or
  Hamilton-product arithmetic.
- `qmul_multicore_compute.cpp` loads Float32 component tiles into FP32
  destination registers and invokes the qmul SFPU product, add-product, and
  subtract-product operations.
- `qmul_sfpu.h` is the only device source containing multiply, add, and
  subtract expressions for the Hamilton product.
- `qmul_multicore_writer.cpp` contains address calculation, CB coordination,
  `noc_async_write_page`, and write barriers only. It contains no SFPU or
  Hamilton-product arithmetic.

## Device and work-split audit

- The host rejects every `--device` value except `0`.
- The host creates exactly one unit mesh with
  `MeshDevice::create_unit_mesh(device_id)`.
- Input component tiles are distributed with
  `split_work_to_cores(grid, component_tiles, true)`, selecting row-major work
  over `min(component_tiles, available_core_count)` cores.
- The N=128 report records one device, device 0, an 8x7 compute grid, 56
  available cores, one component tile, one selected core, planar 32x32 Float32
  layout, and `tensix_compute_sfpu` arithmetic.
- On that measured grid, the official Stage B sizes select multiple cores:

| N | Component tiles | Selected cores |
|---:|---:|---:|
| 4,096 | 4 | 4 |
| 65,536 | 64 | 56 |
| 262,144 | 256 | 56 |

The metrics-v2 validator independently enforces the one-device identity,
component-tile calculation, grid capacity, selected-core calculation, layout,
row-major split, and compute/SFPU arithmetic-path metadata.

## Eligibility decision

The candidate satisfies the Stage B architectural contract: one Wormhole
device, multiple Tensix cores at Stage B sizes, Float32 destination
accumulation, and Hamilton arithmetic confined to the compute/SFPU path.
Changing the implementation constant to `performance_eligible=true` is
therefore authorized. The first performance report must retain
`stable_benchmark=false`; no acceleration claim is authorized by this audit.
