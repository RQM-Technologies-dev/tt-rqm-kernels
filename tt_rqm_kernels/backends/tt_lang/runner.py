"""Subprocess runner for TT-Lang simulator StructuredBench reports."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from tt_rqm_kernels.backends.tt_lang.availability import (
    TTLangSimulatorUnavailable,
    check_tt_lang_sim,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
KERNEL_SCRIPT = REPO_ROOT / "tt_rqm_kernels" / "backends" / "tt_lang" / "qmul_sim_kernel.py"


class QmulCase(Protocol):
    """Minimal case interface consumed by the TT-Lang runner."""

    workload: str
    items: int
    iterations: int
    warmup: int


def run_qmul_cases(
    cases: Iterable[QmulCase],
    *,
    seed: int,
    sim_cli: str | None = None,
) -> dict[str, object]:
    """Run qmul cases through `tt-lang-sim` and combine their reports."""

    availability = check_tt_lang_sim(sim_cli=sim_cli)
    if not availability.available or availability.sim_cli is None:
        raise TTLangSimulatorUnavailable(availability)

    reports = [
        _run_one_case(case, seed=seed + index, sim_cli=availability.sim_cli)
        for index, case in enumerate(cases)
    ]
    if not reports:
        raise ValueError("at least one qmul case is required")

    first = reports[0]
    combined = {
        key: value
        for key, value in first.items()
        if key not in {"generated_at_utc", "results", "seed"}
    }
    combined.update(
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
            "tt_lang_sim": _simulator_metadata(
                first,
                sim_cli=availability.sim_cli,
                version=availability.version,
            ),
            "results": [
                result
                for report in reports
                for result in _report_results(report)
            ],
        }
    )
    return combined


def _simulator_metadata(
    report: dict[str, object],
    *,
    sim_cli: str,
    version: str | None,
) -> dict[str, object]:
    metadata = report.get("tt_lang_sim", {})
    if not isinstance(metadata, dict):
        metadata = {}
    return {
        **metadata,
        "sim_cli": Path(sim_cli).name,
        "sim_version": version,
    }


def _run_one_case(
    case: QmulCase,
    *,
    seed: int,
    sim_cli: str,
) -> dict[str, object]:
    if case.workload != "qmul":
        raise ValueError(f"TT-Lang simulator currently supports qmul only, got {case.workload}")

    with tempfile.TemporaryDirectory(prefix="tt-rqm-ttlang-") as tmp_dir:
        output_path = Path(tmp_dir) / "report.json"
        command = [
            sim_cli,
            str(KERNEL_SCRIPT),
            "--items",
            str(case.items),
            "--iters",
            str(case.iterations),
            "--warmup",
            str(case.warmup),
            "--seed",
            str(seed),
            "--json-output",
            str(output_path),
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=_sim_env(),
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "tt-lang-sim qmul run failed\n"
                f"command: {' '.join(command)}\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        return json.loads(output_path.read_text(encoding="utf-8"))


def _report_results(report: dict[str, object]) -> list[dict[str, object]]:
    results = report["results"]
    if not isinstance(results, list):
        raise TypeError("report results must be a list")
    for result in results:
        if not isinstance(result, dict):
            raise TypeError("each report result must be a dict")
    return results


def _sim_env() -> dict[str, str]:
    env = os.environ.copy()
    repo = str(REPO_ROOT)
    existing = env.get("PYTHONPATH")
    if existing:
        parts = existing.split(os.pathsep)
        if repo not in parts:
            env["PYTHONPATH"] = os.pathsep.join([repo, existing])
    else:
        env["PYTHONPATH"] = repo
    return env
