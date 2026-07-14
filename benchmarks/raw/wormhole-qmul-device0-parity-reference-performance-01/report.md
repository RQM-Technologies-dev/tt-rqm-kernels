# Persistent-device Stage B qmul report

Generated: `2026-07-14T21:37:12.600669+00:00`

Stage: `performance`
Device: `tenstorrent/wormhole-device-0`
Implementation: `multicore_tensix_sfpu_qmul_persistent`
Performance eligible: `true`
Stable benchmark: `false`

Independent persistent Wormhole device-0 performance session wormhole-qmul-device0-parity-reference-performance-01; stability qualification is separate.

## Validated results

| N | values | iters | samples | median device s | p95 device s | max abs error |
|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 16384 | 30 | 10 | 0.001601423 | 0.001626753 | 7.663e-07 |
| 65536 | 262144 | 30 | 10 | 0.002077258 | 0.002146197 | 1.542e-06 |
| 262144 | 1048576 | 30 | 10 | 0.004233331 | 0.004262865 | 1.487e-06 |

## Lifecycle

One host process created Wormhole device 0 once, executed every listed case, and closed it once.
Device 1 was not opened or used.

## Timing contract

The primary elapsed field remains prepared-workload device time. Additive phase records expose device creation, buffer allocation, program build, H2D, prewarm synchronization, warmup, each measured sample, D2H, cleanup, device close, and host process end-to-end time.

This is a first persistent-device hardware sample. It is not a stability result, acceleration claim, or CPU comparison.
