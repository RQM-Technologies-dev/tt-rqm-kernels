# Scientific Kernel Positioning

RQM Technologies is building `tt-rqm-kernels` as a compact structured-kernel
benchmark for Tenstorrent-adjacent scientific and HPC-style workloads. Tenstorrent
is already surfacing non-LLM scientific examples such as spectral element methods
on Wormhole-class hardware; `tt-rqm-kernels` complements that direction with
smaller `[N, 4]` quaternion, rotor, and phase-aware tensor operators that can be
validated against CPU/PyTorch references, scalar spot checks, TT-Lang simulator
output, and future TT-Metalium or TT-NN backends.

## Do Say

- `tt-rqm-kernels` is a structured scientific-kernel benchmark.
- The current proof path is CPU/PyTorch reference output, scalar checks, TT-Lang
  simulator output, and future Tenstorrent backend comparison.
- `qmul` over `[N, 4]` tensors is a compact benchmark for fixed cross-lane
  dependencies, data movement, fusion, register reuse, numerical error, and
  arithmetic-intensity reporting.
- The project complements other Tenstorrent scientific/HPC efforts by focusing
  on a smaller reusable operator class.

## Do Not Say

- Do not say spectral element methods require quaternion kernels.
- Do not claim `tt-rqm-kernels` is integrated with Nekbone or any spectral
  element code.
- Do not imply Tenstorrent endorses RQM Technologies or RQM theory.
- Do not claim hardware performance from CPU/PyTorch or simulator reports.
- Do not frame this as a request for native quaternion hardware or new silicon
  features.

## Narrow Follow-Up Draft

Tenstorrent is already surfacing non-LLM scientific workloads such as spectral
element methods on Wormhole-class hardware. RQM Technologies is approaching the
same ecosystem from a smaller structured-kernel benchmark angle:
`tt-rqm-kernels` uses ordinary floating-point tensors to represent `[N, 4]`
quaternion, rotor, and phase-aware values, with CPU/PyTorch references, scalar
spot checks, a TT-Lang simulator `qmul`, and a future TT-Metalium comparison
path.

The intent is not to claim that spectral element methods need quaternion
kernels. The intent is to contribute a compact benchmark class for structured
scientific tensor operators below applications and above scalar elementwise
math, starting with `qmul` for `[N, 4]` tensors.
