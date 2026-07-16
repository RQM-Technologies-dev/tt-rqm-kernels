# TT-Metalium H2A compensated Candidate B

This is a distinct single-core correctness candidate. It preserves the
original candidate and retained blocker unchanged and remains pre-Claim-Level-0.

The kernel forms `dt/hbar` and each `theta`/`alpha` product as an FP32 high/low
pair. Pinned Wormhole exposes SFPMAD, but a device diagnostic showed that its
product residual is quantized at the relevant scale. The selected path instead
uses Dekker operand splitting with `4097`, which recovered the exact product of
the two FP32 inputs in the frozen large case.

The pair is reduced entirely on device:

```text
reduced = (angle_hi - n * period_hi)
        + (angle_lo - n * period_lo)
```

with split `2π` constants. It becomes one FP32 value only after cancellation
has made the angle small, then feeds pinned `sin_tile` and `cos_tile`. The
ordinary FP32 `r²` accumulation and `sqrt_tile<false>` remain unchanged because
Candidate B already meets the frozen contract. Exact-zero selection still
prevents reciprocal-at-zero and returns `[1,0,0,0]` exactly.

On N300 device 0 the final binary SHA-256 is
`433e74b827d2cf9a7a790a6c9d7bb3917fc1fed3915ec384de0486cdc014d306` and
the final source-bundle SHA-256 is
`7fb65217e05139bf035952ebeb34602d49e5f1772b8dec4c336b7a296e1fba2f`.
All nine development cases and the one non-designated pilot passed. This is a
software arrangement of FP32 operations, not increased hardware precision,
and it makes no stability, performance, acceleration, bandwidth, energy,
dual-device, H2B, or Tenstorrent-endorsement claim.
