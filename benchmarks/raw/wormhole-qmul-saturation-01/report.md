# Persistent-device Stage B qmul report

Generated: `2026-07-14T21:54:54.046487+00:00`

Stage: `diagnostic`
Device: `tenstorrent/wormhole-device-0`
Implementation: `multicore_tensix_sfpu_qmul_persistent`
Performance eligible: `true`
Stable benchmark: `false`

Independent persistent Wormhole device-0 diagnostic session wormhole-qmul-saturation-01; stability qualification is separate.

## Validated results

| N | values | iters | samples | median device s | p95 device s | max abs error |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 4096 | 30 | 10 | 0.001592354 | 0.001610265 | 1.107e-06 |
| 4096 | 16384 | 30 | 10 | 0.001602264 | 0.001686744 | 7.663e-07 |
| 16384 | 65536 | 30 | 10 | 0.001615534 | 0.001734583 | 9.229e-07 |
| 57344 | 229376 | 30 | 10 | 0.001743912 | 0.001813833 | 1.266e-06 |
| 65536 | 262144 | 30 | 10 | 0.002066074 | 0.002145429 | 1.542e-06 |
| 131072 | 524288 | 30 | 10 | 0.002511186 | 0.002548405 | 1.275e-06 |
| 262144 | 1048576 | 30 | 10 | 0.004138704 | 0.004300938 | 1.487e-06 |
| 524288 | 2097152 | 30 | 10 | 0.005661990 | 0.005745803 | 1.474e-06 |
| 1048576 | 4194304 | 30 | 10 | 0.010512042 | 0.010631146 | 1.616e-06 |

## Lifecycle

One host process created Wormhole device 0 once, executed every listed case, and closed it once.
Device 1 was not opened or used.

## Timing contract

The primary elapsed field remains prepared-workload device time. Additive phase records expose device creation, buffer allocation, program build, H2D, prewarm synchronization, warmup, each measured sample, D2H, cleanup, device close, and host process end-to-end time.

This is a first persistent-device hardware sample. It is not a stability result, acceleration claim, or CPU comparison.
