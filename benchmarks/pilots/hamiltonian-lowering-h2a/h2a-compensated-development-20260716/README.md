# H2A compensated-angle development comparison

This non-designated development package compares the immutable original H2A
candidate with a distinct compensated candidate. It is development evidence,
not Claim Level 0 evidence or a performance result.

The exact large case has no `r²` accumulation error. Ordinary FP32 angle
formation loses the decisive information in the large `theta` and `alpha`
products before trigonometry. The first hardware Candidate B used Wormhole
SFPMAD for a product residual; it compiled, but its residual was quantized and
three values still failed. The selected Candidate B instead uses a Dekker
TwoProduct with the exact FP32 splitter `4097`, carries the high/low angle
through a split-`2π` device reduction, and collapses only after reduction.

Corrected `r²` accumulation and corrected square root were not needed. The
selected candidate passed all nine frozen hardware cases and the retained
[non-designated pilot](../h2a-compensated-n300-pilot-20260716/) on N300 device
0. The aggregate and every case remain `stable_benchmark=false`,
`performance_eligible=false`, and `claim_level=null`.

The machine-readable [comparison report](comparison-report.json) contains the
original and compensated identities, two-lane error decomposition, rejected
strategies, and final correctness metrics. The `diagnostics/` directories are
untouched device packages for the SFPMAD product, split product, ordinary and
compensated reductions, trigonometric values, and final failing/passing large
probes.
