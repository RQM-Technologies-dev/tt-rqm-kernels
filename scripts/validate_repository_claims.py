#!/usr/bin/env python3
"""Fail closed when repository status surfaces contradict protected releases."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tt_rqm_kernels.repository_claims import validate_repository_claims


def main() -> int:
    try:
        validate_repository_claims(ROOT)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(str(exc))
        return 1
    print(
        "repository claims valid: qmul Level 2, SU2 fused-only Level 2, H2A pre-hardware foundation"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
