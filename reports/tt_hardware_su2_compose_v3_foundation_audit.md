# SU2ComposeBench v3 Foundation Audit

Date: 2026-07-15
Device: N300, Wormhole device 0
TT-Metal commit: `dd2849b5bc6b7a5d38a9eafbeba31ef8d530f8d4`

## Scope

This is non-designated foundation evidence. It does not qualify Claim Level 2,
does not freeze the v3 preregistration, and does not replace or modify any v2
session. No `su2-v3-level2-session-*` collection was performed.

The v3 change is a host measurement-surface change. The fused reader, compute,
writer, common compute header, and SFPU arithmetic sources are unchanged. The
new candidate mode constructs only the existing fused workload, enqueues its
frozen pilot repeats nonblocking, and uses one `Finish` boundary per raw sample.
The unfused kernels remain available to historical comparison mode and future
Level 3 work but are not constructed or executed in `fused_stability`.

## Build and fused conformance

The candidate compiled successfully with GCC 11.4.0 against the pinned
TT-Metalium CMake package. The binary SHA-256 was:

```text
785fa452345ad340413dae4d59bd0bb3ac0c0398aa81e8f98250cf4a0c82c6a0
```

Fused-only conformance ran in a separate process with CPU affinity `24-27`, a
new empty `TT_METAL_CACHE`, and no unfused output fields. Both cases passed
whole-output validation:

| B | K | fused max absolute error |
|---:|---:|---:|
| 32 | 8 | 1.416818e-7 |
| 2,048 | 8 | 2.969358e-7 |

The device lifecycle recorded one create and one close. The post-run runtime
cache inventory contained 127 files. The conformance session manifest SHA-256
is `9a3abf2dc2636c9ad60eed473f11f64f9573b72bffe82e60280303e9ea3908da`.

## Three non-designated pilots

Three separate collector invocations used the same binary, CPU affinity
`24-27` on the device-local NUMA node, the same provisional repeat plan, and
three distinct empty-start runtime-cache roots. Every report retained ten raw
fused samples after five warmups, whole-output correctness, host state,
pre/post device health, command, stdout/stderr, and cache inventory.

| B | K | repeats | raw duration range | max within-session dispersion | max cross-session deviation |
|---:|---:|---:|---:|---:|---:|
| 32,768 | 8 | 267 | 26.185-26.221 ms | 0.0671% | 0.0045% |
| 8,192 | 32 | 90 | 34.127-34.231 ms | 0.1417% | 0.0378% |
| 2,048 | 128 | 24 | 36.793-36.849 ms | 0.0620% | 0.0139% |
| 512 | 512 | 7 | 43.004-43.073 ms | 0.0742% | 0.0349% |
| 1,024 | 128 | 24 | 36.786-36.844 ms | 0.0553% | 0.0117% |
| 4,096 | 128 | 24 | 36.773-36.854 ms | 0.0720% | 0.0047% |
| 16,384 | 128 | 24 | 36.828-37.077 ms | 0.5124% | 0.0065% |
| 65,536 | 128 | 12 | 36.909-36.988 ms | 0.0724% | 0.0545% |

The deterministic assessor reported:

```text
ready_to_freeze_v3=true
all_cases_within_preferred_5_percent=true
stable_benchmark=false
qualification_passed=false
rejected_gates=[]
```

The pilot manifest SHA-256 values are:

```text
pilot-1  2352ce0b6630d2a8e55417165cbed1be28c6d1f52e88b0eb6370356dbb6278ec
pilot-2  72a1a4cb3e1b8a0ded6ad8b5f3bfcd506e9bdcb2889f37fabf3a8cf25fbd940a
pilot-3  af3359f6748f9493eb3404c8c196d42f186ec3de392cb224fa35f84aee02bfd5
```

The assessment SHA-256 is
`a5ec737d1648fe6acabff4425a16ae51bdaedce35eff674e8999389a174cc1e8`.
The retained remote foundation evidence is under
`/home/user/su2-v3-pilots/` on host `f05cs07`.

## Freeze gate

The pilot data support freezing the measured repeat counts and 5% fused gates,
but the checked-in preregistration intentionally remains
`pilot_foundation_not_frozen`. Before changing it to
`frozen_before_designated_session_1`, commit the v3 implementation so the
candidate can bind a clean execution-source commit and source-tree hash, then
retain or import the pilot evidence at stable repository paths. Only after that
freeze may designated Level 2 collection begin.
