# Persistent-device Stage B qmul report

Generated: `2026-07-14T21:37:59.968291+00:00`

Stage: `diagnostic`
Device: `tenstorrent/wormhole-device-0`
Implementation: `multicore_tensix_sfpu_qmul_persistent`
Performance eligible: `true`
Stable benchmark: `false`

Independent persistent Wormhole device-0 diagnostic session wormhole-qmul-core-scaling-01; stability qualification is separate.

## Validated results

| N | values | iters | samples | median device s | p95 device s | max abs error |
|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 16384 | 30 | 10 | 0.002297206 | 0.002439954 | 7.663e-07 |
| 4096 | 16384 | 30 | 10 | 0.001748616 | 0.001779121 | 7.663e-07 |
| 4096 | 16384 | 30 | 10 | 0.001594303 | 0.001638203 | 7.663e-07 |
| 65536 | 262144 | 30 | 10 | 0.019351237 | 0.020398665 | 1.542e-06 |
| 65536 | 262144 | 30 | 10 | 0.010267207 | 0.010294742 | 1.542e-06 |
| 65536 | 262144 | 30 | 10 | 0.005795249 | 0.005813169 | 1.542e-06 |
| 65536 | 262144 | 30 | 10 | 0.003526013 | 0.003541703 | 1.542e-06 |
| 65536 | 262144 | 30 | 10 | 0.002525058 | 0.002572263 | 1.542e-06 |
| 65536 | 262144 | 30 | 10 | 0.002040949 | 0.002053238 | 1.542e-06 |
| 65536 | 262144 | 30 | 10 | 0.002062114 | 0.002075058 | 1.542e-06 |
| 262144 | 1048576 | 30 | 10 | 0.075392801 | 0.075413515 | 1.487e-06 |
| 262144 | 1048576 | 30 | 10 | 0.037504190 | 0.037945451 | 1.487e-06 |
| 262144 | 1048576 | 30 | 10 | 0.019424040 | 0.019475775 | 1.487e-06 |
| 262144 | 1048576 | 30 | 10 | 0.010386185 | 0.010394921 | 1.487e-06 |
| 262144 | 1048576 | 30 | 10 | 0.005997151 | 0.006007337 | 1.487e-06 |
| 262144 | 1048576 | 30 | 10 | 0.004579932 | 0.004599032 | 1.487e-06 |
| 262144 | 1048576 | 30 | 10 | 0.004199001 | 0.004246246 | 1.487e-06 |

## Lifecycle

One host process created Wormhole device 0 once, executed every listed case, and closed it once.
Device 1 was not opened or used.

## Timing contract

The primary elapsed field remains prepared-workload device time. Additive phase records expose device creation, buffer allocation, program build, H2D, prewarm synchronization, warmup, each measured sample, D2H, cleanup, device close, and host process end-to-end time.

This is a first persistent-device hardware sample. It is not a stability result, acceleration claim, or CPU comparison.
