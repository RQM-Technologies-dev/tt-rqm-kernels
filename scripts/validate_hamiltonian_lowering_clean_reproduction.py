#!/usr/bin/env python3
"""Validate clean-build and clean-tree H2A reproduction evidence."""

from __future__ import annotations

import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tt_rqm_kernels.hamiltonian_lowering_evidence import validate_clean_reproduction


if __name__ == "__main__":
    print(json.dumps(validate_clean_reproduction(REPO), indent=2, sort_keys=True))
