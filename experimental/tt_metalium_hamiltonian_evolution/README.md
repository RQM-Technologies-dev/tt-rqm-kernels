# TT-Metalium H2B device-resident evolution candidate

This source candidate implements H2B as two TT-Metal programs in one Wormhole
device-0 session. It is a device-resident pipeline whose H1 composition stage
is fused; it is not one fused kernel.

```text
host Hamiltonians + dt
  -> one H2D write
  -> compensated H2A program on one Tensix core
  -> six-plane intermediate in device DRAM
  -> protected fused H1 program across available trajectory-tile cores
  -> one final D2H read
```

The candidate creates and closes device 0 once and allocates the input,
intermediate, and final buffers once. H2A writes its output directly into the
same `MeshBuffer` passed to the H1 reader. There is no intermediate
`EnqueueReadMeshBuffer`, no host unpack/repack, and no intermediate
`EnqueueWriteMeshBuffer`.

## Layouts

All device pages are 32x32 FP32 tiles. With
`component_tiles = ceil(B / 1024)`, input and intermediate pages use:

```text
page = (step * 6 + lane) * component_tiles + batch_tile
```

Input lanes are `[h0,hx,hy,hz,dt,inverse_hbar]`. Scalar `dt` is expanded while
packing the public logical input. Intermediate lanes are
`[w,x,y,z,phase_real,phase_imag]`, exactly matching the protected fused H1
reader. Final output uses six component-planar pages and is unpacked only after
the one final device read into logical `[B,4]` and `[B,2]` tensors.

## Protected source reuse

The build compiles the compensated H2A compute source and SFPU header from
`experimental/tt_metalium_hamiltonian_lowering_compensated/` without editing
them. It compiles the existing `su2_fused_reader.cpp`,
`su2_fused_compute.cpp`, `su2_fused_writer.cpp`, `su2_compute_common.h`, and
`su2_sfpu.h` from `experimental/tt_metalium_su2_compose/` without editing them.
Only the H2B H2A reader/writer are new, because H2B requires step-major pages.

Build against the pinned baseline:

```bash
python experimental/tt_metalium_hamiltonian_evolution/check_environment.py \
  --tt-metal-root /path/to/tt-metal
python experimental/tt_metalium_hamiltonian_evolution/build_candidate.py \
  --tt-metal-root /path/to/tt-metal \
  --cmake-prefix-path /path/to/tt-metal/build_Release
```

The pinned TT-Metal commit is
`dd2849b5bc6b7a5d38a9eafbeba31ef8d530f8d4`.

This is a candidate source foundation. Until a retained pilot passes, hardware
has not been run. It is `stable_benchmark=false`,
`performance_eligible=false`, and `claim_level=null`; it makes no performance,
stability, acceleration, bandwidth, energy, dual-device, or endorsement claim.
