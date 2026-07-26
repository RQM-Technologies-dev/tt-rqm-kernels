# Draft update for tenstorrent/tt-metal#49887

Do not post without explicit authorization.

We prepared a small current-main programming-example form of the qmul work as a
concrete placement candidate under
`tt_metal/programming_examples/contributed/qmul/`.

It preserves the ordinary FP32 `[N,4]` representation, `[real,i,j,k]` lane
order, and noncommutative Hamilton product `a * b`. The example is
self-contained: deterministic C++ inputs, AoS-to-planar tile conversion,
multicore reader/compute/writer kernels, Float64 whole-output validation, and
clean single-device closure. Hamilton arithmetic remains in the Tensix/SFPU
compute kernel; the reader and writer are data movement only.

The focused source checks and narrow current-main Linux target build pass. On a
configured N300, single runs at `N=128` and `N=4096` validated every logical
output with zero failures or nonfinite values and closed the device cleanly.
The exact nonclaiming results and provenance are recorded in the accompanying
port note.

If `contributed` is not the preferred surface, RQM can relocate the same
contract to an experimental TT-NN operation with maintainer guidance. This port
does not inherit the prior qmul benchmark qualification or make an acceleration
claim.
