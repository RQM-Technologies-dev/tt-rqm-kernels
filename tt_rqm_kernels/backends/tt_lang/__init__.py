"""Optional TT-Lang simulator backend support.

This package is intentionally optional. The default `tt-rqm-kernels`
installation does not depend on TT-Lang; users opt in by installing
`tt-lang-sim` in an isolated environment.
"""

from tt_rqm_kernels.backends.tt_lang.availability import (
    SETUP_HINT,
    TTLangAvailability,
    TTLangSimulatorUnavailable,
    check_tt_lang_sim,
)

name = "tt-lang-sim"

__all__ = [
    "SETUP_HINT",
    "TTLangAvailability",
    "TTLangSimulatorUnavailable",
    "check_tt_lang_sim",
    "name",
]
