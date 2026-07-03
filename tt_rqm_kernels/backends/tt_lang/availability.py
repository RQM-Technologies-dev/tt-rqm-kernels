"""Availability checks for the optional TT-Lang simulator backend."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

SETUP_HINT = """Install TT-Lang simulation in an isolated Python 3.12 environment:

python3.12 -m venv --prompt ttlang ttlang-venv
source ttlang-venv/bin/activate
python -m pip install tt-lang-sim
tt-lang-setup
python scripts/run_ttlang_qmul_smoke.py --items 128
"""


@dataclass(frozen=True)
class TTLangAvailability:
    """Detected TT-Lang simulator state."""

    available: bool
    sim_cli: str | None
    version: str | None
    reason: str
    setup_hint: str = SETUP_HINT


def check_tt_lang_sim(*, sim_cli: str | None = None) -> TTLangAvailability:
    """Return whether the `tt-lang-sim` console command is available.

    The simulator injects its own `ttl` and `ttnn` modules when it executes a
    script, so the CLI is the source of truth here. A plain Python import check
    would incorrectly reject valid simulator environments.
    """

    resolved_cli = _resolve_cli(sim_cli)
    if resolved_cli is None:
        return TTLangAvailability(
            available=False,
            sim_cli=None,
            version=None,
            reason="tt-lang-sim was not found on PATH.",
        )

    version = _read_version(resolved_cli)
    return TTLangAvailability(
        available=True,
        sim_cli=resolved_cli,
        version=version,
        reason="tt-lang-sim CLI is available.",
    )


def _read_version(sim_cli: str) -> str | None:
    try:
        completed = subprocess.run(
            [sim_cli, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    output = (completed.stdout or completed.stderr).strip()
    return output or None


def _resolve_cli(sim_cli: str | None) -> str | None:
    if sim_cli is None:
        return shutil.which("tt-lang-sim")
    if "/" in sim_cli:
        path = Path(sim_cli)
        return str(path) if path.exists() and os.access(path, os.X_OK) else None
    return shutil.which(sim_cli)
