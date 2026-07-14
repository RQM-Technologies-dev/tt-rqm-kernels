# Persistent-device Stage B qmul timing audit

Date: 2026-07-14

This audit covers the first persistent-device performance sample. It verifies
timer ownership, synchronization boundaries, lifecycle counts, and report
arithmetic. It does not set `stable_benchmark=true` and does not make an
acceleration or CPU-comparison claim.

## Session identity

- Protocol: `tt-rqm-external-qmul-persistent.v1`
- Metrics: `tt-rqm-external-qmul-persistent-metrics.v1`
- Implementation: `multicore_tensix_sfpu_qmul_persistent`
- Execution-source commit: `3ae68815e8ac025e49f09d3797dbbac2f77245b3`
- Candidate SHA-256: `179a5cc3e6b146a1e8c61e61ab9ab173bbc543f88181b91c8621a7e959c98ce5`
- TT-Metalium commit: `dd2849b5bc6b7a5d38a9eafbeba31ef8d530f8d4`
- Hardware scope: one Wormhole unit mesh, logical device 0
- Lifecycle: one process, one device create, one device close

The report hash matches `sha256sum` of the executed binary. The candidate ran
all three performance cases before the single close. Device 1 was never opened.

## Timer ownership and synchronization

| Field | Owner | Start / stop | Completion boundary |
|---|---|---|---|
| host process end-to-end | Python harness `perf_counter` | immediately before/after the one subprocess | process exit |
| candidate session | C++ steady clock | persistent `main` entry through clean device close, before metrics serialization | device close complete |
| device create | C++ steady clock | around `MeshDevice::create_unit_mesh(0)` | API return |
| buffer allocation | C++ steady clock, per size | around three replicated DRAM buffer creations | API returns |
| program build | C++ steady clock, per size | around audited workload construction and runtime args | workload construction return |
| H2D | C++ steady clock, per size | before two writes | explicit `Finish` after both writes |
| prewarm sync | C++ steady clock, per size | after H2D and program build | explicit `Finish`; no hidden measured workload |
| warmup | C++ steady clock, per size | before five workload enqueues | explicit `Finish`; excluded from samples |
| measured sample | C++ steady clock, ten per size | before 30 prepared-workload enqueues | explicit `Finish`; primary elapsed source |
| D2H | C++ steady clock, per size | before output read | blocking read plus explicit `Finish` |
| cleanup | C++ steady clock, per size | around buffer releases | releases returned |
| device close | C++ steady clock | around the one mesh close | close returned successfully |

Every device-facing phase has a completion boundary. The primary `elapsed_s`
retains the existing meaning of prepared-workload enqueue plus completion; the
new report adds phase and lifecycle fields rather than redefining it.

## Raw sample summary

The JSON report retains all 30 raw measured values. Medians use the average of
the two central sorted values; p95 uses nearest rank.

| N | iterations/sample | samples | median s | nearest-rank p95 s | p95 spread | preregistered limit |
|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 30 | 10 | 0.0016516275 | 0.001724061 | 4.3856% | 10.4825% |
| 65536 | 30 | 10 | 0.0021010870 | 0.002143976 | 2.0413% | 5.0000% |
| 262144 | 30 | 10 | 0.0042311940 | 0.004303374 | 1.7059% | 5.0000% |

Each within-session spread is below the preregistered size-specific limit.
That observation does not satisfy the independent three-session requirement,
so `stable_benchmark` correctly remains `false`.

## Session accounting

- Candidate session: `2.407865554 s`
- Device create: `2.087592366 s`
- Device close: `0.130627294 s`
- Host process end-to-end: `3.06052512700262 s`
- Accounted N=4096 phases: `0.047639504 s`
- Accounted N=65536 phases: `0.024977350 s`
- Accounted N=262144 phases: `0.051390014 s`

The sum of nested phase timers is less than the enclosing candidate-session
timer, and that timer is less than the independently measured host-process
timer. Unaccounted candidate time includes host input reads and planar
conversion, output conversion and file writes, metrics construction, and
ordinary control-flow gaps. Host-only overhead additionally includes process
startup/teardown and Python output validation.

## Correctness and health gate

All 1,327,104 Float32 outputs across the three sizes were compared to an
independent Float64 Hamilton-product golden result. There were zero failing and
zero non-finite values at `atol=1e-4`, `rtol=1e-4`. Work metadata selected 4,
56, and 56 cores, respectively, on the 8x7 grid. Before and after the session,
both N300 sides reported healthy DRAM and `FAULTS=0x0`; the post-run snapshot
also reported `THROTTLER=0x0` on both sides.

## Audit conclusion

The artifact is a valid first persistent-device Stage B timing sample with
complete lifecycle, phase, raw-sample, output-identity, and provenance
records. It supports continued stability work under the preregistered policy.
It does not support a stable-performance label, acceleration claim, CPU
comparison, dual-device conclusion, or second-side result.
