# Persistent-device Stage B qmul report

Generated: `2026-07-14T21:38:40.888625+00:00`

Stage: `diagnostic`
Device: `tenstorrent/wormhole-device-0`
Implementation: `multicore_tensix_sfpu_qmul_persistent`
Performance eligible: `true`
Stable benchmark: `false`

Independent persistent Wormhole device-0 diagnostic session wormhole-qmul-initialization-order-b-01; stability qualification is separate.

## Validated results

| N | values | iters | samples | median device s | p95 device s | max abs error |
|---:|---:|---:|---:|---:|---:|---:|
| 262144 | 1048576 | 30 | 10 | 0.004228976 | 0.004267305 | 1.487e-06 |
| 65536 | 262144 | 30 | 10 | 0.001902695 | 0.001917319 | 1.542e-06 |
| 4096 | 16384 | 30 | 10 | 0.001323296 | 0.001340046 | 7.663e-07 |

## Lifecycle

One host process created Wormhole device 0 once, executed every listed case, and closed it once.
Device 1 was not opened or used.

## Timing contract

The primary elapsed field remains prepared-workload device time. Additive phase records expose device creation, buffer allocation, program build, H2D, prewarm synchronization, warmup, each measured sample, D2H, cleanup, device close, and host process end-to-end time.

This is a first persistent-device hardware sample. It is not a stability result, acceleration claim, or CPU comparison.
