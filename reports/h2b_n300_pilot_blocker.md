# H2B N300 pilot blocker

The first non-designated H2B N300 pilot is retained and did not pass. The failure is classified as `environment`.

All 20 frozen cases were invoked once, in order, without retry or replacement. Each stopped during TT-Metal runtime initialization before device execution because this build required `TT_METAL_RUNTIME_ROOT`; the frozen launcher supplied `TT_METAL_HOME` but not that separate runtime-root variable. No final rotor, phase, or matrix output was produced, so this pilot contains no case-level numerical result and no numerical failure.

The source commit, source bundle, candidate binary, and pinned TT-Metal identities did not change. Both N300 device entries remained visible in the retained pre-run and post-run health records.

Session 1 must remain immutable. Any future non-designated attempt requires a newly versioned and newly frozen contract that explicitly binds the runtime-root environment. There is no designated contract and no H2B hardware claim.
