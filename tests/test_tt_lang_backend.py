from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tt_rqm_kernels.backends.tt_lang.availability import check_tt_lang_sim
from tt_rqm_kernels.structuredbench import SCHEMA_VERSION


def test_tt_lang_availability_reports_missing_cli() -> None:
    availability = check_tt_lang_sim(sim_cli="/path/that/does/not/exist/tt-lang-sim")

    assert availability.available is False
    assert availability.sim_cli is None
    assert (
        availability.reason
        == "requested tt-lang-sim CLI is not executable: /path/that/does/not/exist/tt-lang-sim"
    )


def test_tt_lang_smoke_check_is_non_mutating() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_ttlang_qmul_smoke.py",
            "--check",
            "--sim-cli",
            "/path/that/does/not/exist/tt-lang-sim",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["available"] is False
    assert payload["sim_cli"] is None
    assert "requested tt-lang-sim CLI is not executable" in payload["reason"]
    assert "setup_hint" in payload


def test_tt_lang_smoke_missing_cli_exits_without_traceback() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_ttlang_qmul_smoke.py",
            "--sim-cli",
            "/path/that/does/not/exist/tt-lang-sim",
            "--items",
            "32",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "TT-Lang simulator unavailable" in completed.stderr
    assert "requested tt-lang-sim CLI is not executable" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_structuredbench_tt_lang_missing_cli_exits_without_traceback() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tt_rqm_kernels.structuredbench",
            "--backend",
            "tt-lang-sim",
            "--suite",
            "qmul",
            "--sim-cli",
            "/path/that/does/not/exist/tt-lang-sim",
            "--items",
            "32",
            "--iters",
            "1",
            "--warmup",
            "0",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "TT-Lang simulator unavailable" in completed.stderr
    assert "requested tt-lang-sim CLI is not executable" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_committed_tt_lang_report_schema_and_claims() -> None:
    report_path = Path("reports/tt_lang_qmul_sim.json")
    markdown_path = Path("reports/tt_lang_qmul_sim.md")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema"] == SCHEMA_VERSION
    assert report["backend"] == "tt-lang-sim"
    assert report["device"] == "functional-simulator"
    assert report["simulation"] is True
    assert report["suite"] == "qmul"
    assert report["dtype"] == "float32"
    assert report["seed"] == 0
    assert "tt_lang_sim" in report

    result = report["results"][0]
    required_result_fields = {
        "arithmetic_intensity_flops_per_byte",
        "checksum",
        "effective_gb_per_s",
        "elapsed_s",
        "estimated_flops",
        "estimated_flops_per_s",
        "items",
        "iterations",
        "latency_ms",
        "max_abs_error",
        "max_rel_error",
        "rms_error",
        "scalar_reference_max_abs_error",
        "structured_shape",
        "throughput",
        "warmup",
        "workload",
    }
    assert required_result_fields <= set(result)
    assert result["structured_shape"] == "[128, 4]"
    assert result["scalar_reference_max_abs_error"] < 1e-4

    markdown = markdown_path.read_text(encoding="utf-8")
    assert "TT-Lang functional simulator" in markdown
    assert "not hardware performance" in markdown
    assert "simulator smoke output" in markdown


@pytest.mark.skipif(
    os.environ.get("TT_RQM_RUN_TTLANG") != "1",
    reason="TT-Lang simulator integration is opt-in.",
)
def test_tt_lang_qmul_smoke_runs_when_opted_in(tmp_path: Path) -> None:
    json_output = tmp_path / "tt_lang_qmul_sim.json"
    markdown_output = tmp_path / "tt_lang_qmul_sim.md"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_ttlang_qmul_smoke.py",
            "--items",
            "32",
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    report = json.loads(json_output.read_text(encoding="utf-8"))
    assert report["schema"] == SCHEMA_VERSION
    assert report["backend"] == "tt-lang-sim"
    assert report["simulation"] is True
    assert report["tt_lang_sim"]["sim_cli"]
    assert "sim_version" in report["tt_lang_sim"]
    assert report["results"][0]["max_abs_error"] < 1e-4
    assert "TT-Lang functional simulator" in markdown_output.read_text(encoding="utf-8")


@pytest.mark.skipif(
    os.environ.get("TT_RQM_RUN_TTLANG") != "1",
    reason="TT-Lang simulator integration is opt-in.",
)
def test_structuredbench_tt_lang_backend_when_opted_in() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tt_rqm_kernels.structuredbench",
            "--backend",
            "tt-lang-sim",
            "--suite",
            "qmul",
            "--items",
            "32",
            "--iters",
            "1",
            "--warmup",
            "0",
            "--format",
            "json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)

    assert report["schema"] == SCHEMA_VERSION
    assert report["backend"] == "tt-lang-sim"
    assert report["simulation"] is True
    assert report["tt_lang_sim"]["sim_cli"]
    assert "sim_version" in report["tt_lang_sim"]
    assert len(report["results"]) == 1
    assert report["results"][0]["items"] == 32
