#!/usr/bin/env python3
"""Write the exact deterministic H2B pilot inputs before hardware execution."""

from __future__ import annotations

import json
from pathlib import Path

from tt_rqm_kernels.hamiltonian_evolution_pilot_contract import (
    DEFAULT_INPUT_ROOT,
    write_frozen_inputs,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    manifest = write_frozen_inputs(ROOT / DEFAULT_INPUT_ROOT)
    print(json.dumps({"case_order": manifest["case_order"], "case_count": len(manifest["cases"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
