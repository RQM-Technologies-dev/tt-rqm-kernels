# Persistent-device Stage B qmul report

Generated: `2026-07-14T21:37:04.943834+00:00`

Stage: `conformance`
Device: `tenstorrent/wormhole-device-0`
Implementation: `multicore_tensix_sfpu_qmul_persistent`
Performance eligible: `true`
Stable benchmark: `false`

Independent persistent Wormhole device-0 conformance session wormhole-qmul-device0-parity-reference-conformance-01; stability qualification is separate.

## Validated results

| N | values | iters | samples | median device s | p95 device s | max abs error |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 512 | 1 | 1 | 0.567513819 | 0.567513819 | 6.606e-07 |

## Lifecycle

One host process created Wormhole device 0 once, executed every listed case, and closed it once.
Device 1 was not opened or used.

## Timing contract

The primary elapsed field remains prepared-workload device time. Additive phase records expose device creation, buffer allocation, program build, H2D, prewarm synchronization, warmup, each measured sample, D2H, cleanup, device close, and host process end-to-end time.

This is a first persistent-device hardware sample. It is not a stability result, acceleration claim, or CPU comparison.
