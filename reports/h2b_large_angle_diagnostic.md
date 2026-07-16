# H2B large-angle development diagnostic

This is a deterministic development report, not release evidence.

## Diagnosis

The retained source-foundation failure originates in uncompensated Float32 angle-product formation before trigonometric reduction. H1 composition, order, layout, synchronization, and nonfinite arithmetic are not the primary source.

The H2B hardware source already uses the protected compensated H2A Dekker TwoProduct and split-2pi reduction. Composing retained compensated H2A N300 step outputs in exact H1 order produces:

- rotor max absolute error: `0.000104454171875`
- phase max absolute error: `1.06862116156e-07`
- failing values at atol=rtol=1e-4: `0`
- complex128 final-matrix error: `0.000140898398484`

This resolves the original failure mechanism, but the wider mixed-direction sweep finds a separate magnitude-formation boundary. The pilot therefore selects acceptance path B.

Frozen pilot domain: `abs(theta) <= 1024.0` and `abs(alpha) <= 8192.0` radians for every logical step. The public mathematical API remains valid outside this implementation conformance domain.

## Sweep

The deterministic sweep contains `166` cases spanning signs, h0, vector magnitude, direction, dt, step count, commuting axes, noncommuting axes, integer/half-integer pi neighborhoods, 2pi multiples, quotient boundaries, and cancellation-sensitive reductions.

Maximum compensated CPU-equivalent rotor error: `0.00068694794388`
Maximum compensated CPU-equivalent phase error: `7.00246775009e-08`
Maximum compensated CPU-equivalent matrix error: `0.000686961883727`

The machine-readable report contains every stage, arithmetic trace, quotient, reduced angle, trigonometric error, and sweep result.
