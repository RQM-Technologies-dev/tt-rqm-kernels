# Persistent-device Stage B qmul report

Generated: `2026-07-14T21:38:47.694012+00:00`

Stage: `diagnostic`
Device: `tenstorrent/wormhole-device-0`
Implementation: `multicore_tensix_sfpu_qmul_persistent`
Performance eligible: `true`
Stable benchmark: `false`

Independent persistent Wormhole device-0 diagnostic session wormhole-qmul-initialization-order-c-01; stability qualification is separate.

## Validated results

| N | values | iters | samples | median device s | p95 device s | max abs error |
|---:|---:|---:|---:|---:|---:|---:|
| 65536 | 262144 | 30 | 10 | 0.002063753 | 0.002080598 | 1.542e-06 |
| 65536 | 262144 | 30 | 10 | 0.002054904 | 0.002070528 | 1.542e-06 |

## Lifecycle

One host process created Wormhole device 0 once, executed every listed case, and closed it once.
Device 1 was not opened or used.

## Timing contract

The primary elapsed field remains prepared-workload device time. Additive phase records expose device creation, buffer allocation, program build, H2D, prewarm synchronization, warmup, each measured sample, D2H, cleanup, device close, and host process end-to-end time.

This is a first persistent-device hardware sample. It is not a stability result, acceleration claim, or CPU comparison.
