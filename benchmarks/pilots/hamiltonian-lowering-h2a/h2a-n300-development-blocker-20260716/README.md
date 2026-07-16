# H2A N300 development blocker — 2026-07-16

This is retained development-probe evidence, not a pilot, designated run, or
release. The real single-core candidate compiled and passed both a focused
hardware smoke and the exact mixed-zero/nonzero case. It then failed the frozen
large-angle gate, so the declared nine-case pilot did not start.

The large-angle maximum errors were:

- rotor: `9.357817553457826e-04`;
- phase: `4.34147712645494e-04`;
- complex matrix reconstruction: `1.2431001873617557e-03`; and
- 8 values outside the frozen `1e-4` tolerance, with zero nonfinite values.

These errors exactly match the existing PyTorch FP32 lowering path for that
case. The pinned Wormhole sine and cosine primitives already use four-stage
Cody–Waite range reduction. The observed blocker is therefore FP32 magnitude
and angle formation at the frozen coefficient scale, not absent SFPU range
reduction. No clamp, tolerance change, host-precomputed angle, revised
preregistration, or substitute pilot was used.

H2A remains without a claim level. `stable_benchmark=false`,
`performance_eligible=false`, and `qualification_eligible=false` remain fixed.
