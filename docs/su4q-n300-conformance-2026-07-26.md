# `su4q-n300-conformance`

Status: **passed, internal correctness evidence only**

This experiment exercised proof-gated `su4q` compiler output on N300 device 0 using a complete quaternion–Cartan implementation. It is not a performance benchmark or an acceleration claim.

## Pinned provenance

- `rqm-compiler`: `38408fac8bec95f5e243727395161360af5223b6`
- adaptive-routing change contained by compiler pin: `ae89a3e`
- `rqm-entanglement`: `b5fa69f6a57eccc0f43765341cf84203c790920a`
- `tt-rqm-kernels` base: `fffa30784a00656a1a26ee89406633fecc9574ed`
- experiment execution source: `c1cfa884de17e5afcbab0ab5a7b3266e5a5c834e`
- `tt-metal`: `9802b80464cfd213b1146189753d8aedf86193fe`

The candidate binary and source-bundle hashes are recorded in each run's `metrics.json`; all input hashes and compiler-routing evidence are in each `manifest.json`.

## Compiler routing

Six deterministic source-circuit windows were selected by `AdaptiveCartanPolicy(mode="selective")`, emitted as proof-verified `su4q`, and serialized from the actual `QuaternionCartanBlock`: RZZ-heavy, XX/YY/ZZ mixed, CNOT-class, mixed-Hamiltonian, asymmetric-local-shell, and generic-interior cases.

The host routing contract also passed safe-mode zero KAK, exact inverse cancellation, low-benefit rejection, changing-pair ineligibility, and selective fallback preservation. A mixed CNOT/iSWAP source window did not pass the compiler's final proof and therefore remained a compiler fallback; iSWAP convention coverage on the device came from a direct structured sentinel. That fallback is not counted as hardware success.

## Device correctness

The device consumed FP32 `[N,20]` blocks and FP32 `[N,8]` states, retained intermediates in device DRAM, and applied right `SU(2)(q1) tensor SU(2)(q0)`, positive-sign Cartan evolution, left local shells, and stored global phase. Validation used unaligned output against an independent complex128 source-circuit oracle.

| N | failed values | nonfinite values | max absolute error | max norm drift | case coverage | clean close |
|---:|---:|---:|---:|---:|---:|:---:|
| 128 | 0 | 0 | `1.7308668465165766e-7` | `1.3913574670176843e-7` | 11/11 | yes |
| 4096 | 0 | 0 | `2.8242813776557796e-7` | `2.120702573549238e-7` | 11/11 | yes |

Both runs passed `atol=1e-4`, `rtol=1e-4`, input/provenance checks, full case representation, and clean device closure. Postflight health showed DRAM healthy and `FAULTS=0x0` on both halves of the N300. Device 1 was not used.

## Verification

- Source-contract, serialization, malformed-input, factor-order, sign, and host-protocol tests: 11 passed.
- Python static checks: passed.
- Narrow TT-Metalium host target: built with `-Wall -Wextra -Werror`.
- Device order: `N=128` passed before `N=4096` was started.

The first `N=128` attempt stopped during device-kernel JIT because the shared header omitted the TT-Metal unary initialization declaration. No candidate output was produced; the device closed cleanly. Commit `c1cfa88` added the required include, and both reported sessions then ran with fresh kernel caches.

## Evidence

Machine-readable evidence is under `benchmarks/pilots/su4q-n300-conformance/`:

- `session-manifest.json`
- sanitized `preflight.json` and `postflight.json`
- per-run `manifest.json`, `metrics.json`, `validation.json`, input/reference binaries, device output, and hashes

This branch remains private and local. No branch was pushed, no issue or Discord post was updated, and no performance sweep was run.
