# TT-Metalium H2A single-core candidate

This directory preserves the original real, compilable TT-Metalium correctness
candidate for H2A coefficient lowering. It is historical experimental source,
not the compensated candidate used by the later Claim Level 0 release and not
a performance benchmark.

## Frozen environment and architecture

- TT-Metal commit: `dd2849b5bc6b7a5d38a9eafbeba31ef8d530f8d4`
- device/core: Wormhole device 0, Tensix core `(0,0)`
- input: component-planar FP32 tiles for `h0,hx,hy,hz,dt,1/hbar`
- output: component-planar FP32 tiles for `w,x,y,z,phase.real,phase.imag`
- logical order: flattened row-major `[B,K]`, deterministically unpadded by the host

The reader and writer perform DMA only. The compute kernel forms `dt/hbar`,
`r²`, `r`, the rotor angle, scalar phase angle, and all six outputs on device.
The host expands scalar `dt` and a scalar `1/hbar` plane but does not calculate
angles, trigonometry, magnitude, reciprocal, rotors, or phases.

## Exact-zero path

The kernel computes `zero_mask = eqz(r²)`, then uses a custom FP32 SFPI lane
selection because pinned `where_tile` supports integer formats, not FP32. It
selects `safe_r=1` in zero lanes before calling `recip_tile<false>`, so the
reciprocal never receives a zero denominator. A second selection writes the
exact identity rotor in those lanes. There is no epsilon threshold, clamp, or
output normalization.

## Pinned primitives

The compute path uses `sqrt_tile<false>`, `recip_tile<false>`, `sin_tile`,
`cos_tile`, `eqz_tile`, tile copy/pack, and repository-style custom SFPI FP32
multiply/add/select/negate functions. FP32 destination accumulation is enabled
with `math_approx_mode=false`. The pinned Wormhole sine and cosine
implementations already perform four-stage Cody–Waite range reduction; the
candidate adds no second range reducer and never reduces angles on the host.

## Build and protocol execution

```bash
python experimental/tt_metalium_hamiltonian_lowering/check_environment.py \
  --tt-metal-root /path/to/tt-metal

python experimental/tt_metalium_hamiltonian_lowering/build_candidate.py \
  --tt-metal-root /path/to/tt-metal \
  --cmake-prefix-path /path/to/tt-metal/build_n300

TT_METAL_HOME=/path/to/tt-metal \
python scripts/validate_hamiltonian_lowering_candidate.py \
  --execution-label hardware \
  --command "python experimental/tt_metalium_hamiltonian_lowering/run_candidate.py"
```

`run_candidate.py` verifies both serialized input SHA-256 values before device
execution and binds metrics to binary, source bundle, Git, compiler, runtime,
TT-Metal, device, core, layout, primitive, and lifecycle metadata.

## Development result and stop gate

On N300 device 0, the candidate binary compiled and passed the focused mixed
zero/nonzero probe with zero nonfinite values and maximum rotor error
`4.2046909509707575e-08`. The exact frozen large-angle probe failed with:

- maximum rotor error `9.357817553457826e-04`;
- maximum phase error `4.34147712645494e-04`;
- matrix reconstruction error `1.2431001873617557e-03`; and
- 12 values outside tolerance.

Those errors exactly match the existing PyTorch FP32 path, identifying FP32
magnitude/angle formation at the frozen scale as the blocker rather than
missing SFPU range reduction. In accordance with the preregistered working
order, the nine-case non-designated pilot did not start. The retained blocker
is under `benchmarks/pilots/hamiltonian-lowering-h2a/`.

This original candidate therefore did not qualify. A later, separately
identified compensated candidate resolved the large-angle precision blocker
and supports the public H2A Claim Level 0 silicon-conformance release. That
release still has no performance eligibility or stable benchmark. H2B remains
out of scope.
