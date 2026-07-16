# TT-Metalium H2A Design Scaffold

This directory is an implementation design and pinned-API audit boundary. It
does not contain an executable TT-Metal candidate and is not hardware evidence.
H2A remains `pre_hardware`, `performance_eligible=false`, and
`stable_benchmark=false`.

## Pinned context

- TT-Metal commit: `dd2849b5bc6b7a5d38a9eafbeba31ef8d530f8d4`
- Target: one Wormhole device, conformance first
- Logical input: FP32 `[B,K,4]` coefficient lanes `[h0,hx,hy,hz]` plus scalar
  or broadcastable `[B,K]` FP32 `dt`
- Logical output: FP32 `[B,K,4]` rotor lanes `[w,x,y,z]` and `[B,K,2]` phase
  lanes `[real,imag]`

The pinned source was inspected on the N300 host. The audit script reproduces
the commit and header checks when a matching TT-Metal checkout is available:

```bash
python experimental/tt_metalium_hamiltonian_lowering/audit_pinned_tt_metal.py \
  --tt-metal-root /path/to/tt-metal
```

## Primitive classification

| Requirement | Classification | Pinned evidence and restriction |
|---|---|---|
| FP32 tile add/multiply | verified available | `api/compute/eltwise_binary.h` exposes `add_tiles` and `mul_tiles` |
| square root | available with restrictions | `eltwise_unary/sqrt.h` exposes compute-engine `sqrt_tile`; DST must be acquired and the implementation uses SFPU approximation controls |
| reciprocal | available with restrictions | `eltwise_unary/recip.h` exposes `recip_tile`; documentation names FP32 as a full-accuracy format, but zero must be masked before use |
| reciprocal square root | available with restrictions | `eltwise_unary/rsqrt.h` exposes templated `rsqrt_tile`; architecture/legacy/fast-approximation behavior must be frozen by the candidate |
| sine and cosine | available with restrictions | `eltwise_unary/trigonometry.h` exposes compute-engine `sin_tile` and `cos_tile` with approximate eight-iteration SFPU calls |
| exact-zero comparison and select | available with restrictions | `eltwise_unary/comp.h` exposes `eqz_tile`; `eltwise_unary/where.h` exposes `where_tile`; their composed FP32 mask behavior still requires candidate conformance |
| vector magnitude | requires approximation or composed implementation | square `hx/hy/hz`, add three planes, then apply `sqrt_tile` or an explicitly frozen `rsqrt_tile` route |
| large-angle range behavior | not yet verified | API presence does not establish accuracy for H2A's large finite angles; hardware conformance must cover them |

This classification is commit-specific. It is not a compatibility claim for a
newer TT-Metal checkout.

## Data and core plan

Flatten logical `[B,K]` to `N=B*K`, pad `N` to complete FP32 tiles, and pack
each component into a separate plane. A scalar `dt` may be represented as a
constant tile; broadcast `dt` is packed as a fifth input plane. Contiguous tile
ranges are distributed row-major over active Tensix cores. Padding lanes are
zeroed and never included in validation or checksums.

The first design uses circular buffers for:

1. `h0`, `hx`, `hy`, `hz`, and `dt` inputs;
2. squared-vector accumulation and magnitude/reciprocal intermediates;
3. `theta`, `alpha`, sine, and cosine intermediates; and
4. six output planes `w`, `x`, `y`, `z`, `phase_real`, `phase_imag`.

Exact buffer indices and depths must be frozen only with compilable source and
an L1 allocation audit; this document does not invent them.

## Reader, compute, and writer responsibilities

- Reader: DMA deterministic contiguous tile ranges, expand scalar `dt` if
  chosen by the implementation, and expose the valid-lane count for the final
  tile.
- Compute: form `r^2`, create an exact-zero mask, compute magnitude and safe
  reciprocal only on nonzero lanes, evaluate `theta`/`alpha`, sine/cosine, and
  select identity rotor lanes at zero. All outputs and intermediates are FP32.
- Writer: store six planar output streams in deterministic flattened order and
  omit padded lanes at the host protocol boundary.

The zero branch must avoid evaluating or consuming `1/r` for zero lanes. No
epsilon threshold, hidden clamp, or normalization is allowed.

## Validation sequence

1. Pass the CPU external-candidate protocol with the reference executable.
2. Implement a standalone TT-Metal H2A candidate in a clean source tree.
3. Freeze source, binary, compiler, runtime, and pinned TT-Metal identities.
4. Run Claim Level 0 only under the H2A preregistration: whole-output checks for
   zero, axis, tiny-norm, random, varying-`dt`, large-angle, and mixed cases.
5. Capture profiler evidence only after correctness. API presence does not
   substitute for hardware validation.

No upstream port or PR begins while
[`tenstorrent/tt-metal#49887`](https://github.com/tenstorrent/tt-metal/issues/49887)
remains unanswered.

## H1 and future H2B boundary

H2A emits the existing logical H1 inputs. H2B may later keep those six planes
device-resident and feed the protected fused H1 ordered-composition kernel.
That fusion requires H2A correctness, H1 compatibility on the selected
TT-Metal baseline, exact `K-1 ... 0` order, and a new claim contract. It cannot
reuse H1's historical stable label.
