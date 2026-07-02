from __future__ import annotations

import json
import subprocess
import sys

from tt_rqm_kernels.structuredbench import (
    SCHEMA_VERSION,
    build_cases,
    render_table,
    run_suite,
)


def test_build_smoke_cases_cover_core_workloads() -> None:
    workloads = {case.workload for case in build_cases("smoke")}

    assert workloads == {"qmul", "qrotate", "qnormalize", "qinverse", "phase_update"}


def test_run_suite_smoke_with_small_overrides() -> None:
    report = run_suite(
        "smoke",
        items_override=32,
        iterations_override=1,
        warmup_override=0,
    )

    assert report["schema"] == SCHEMA_VERSION
    assert report["suite"] == "smoke"
    assert report["backend"] == "torch"
    assert len(report["results"]) == 5
    for result in report["results"]:
        assert result["items"] == 32
        assert result["iterations"] == 1
        assert result["throughput"] > 0.0
        assert result["max_abs_error"] >= 0.0


def test_render_table_includes_workloads() -> None:
    report = run_suite("qmul", items_override=16, iterations_override=1, warmup_override=0)
    table = render_table(report)

    assert "StructuredBench" in table
    assert "schema=structuredbench.v1" in table
    assert "qmul" in table


def test_module_cli_json_smoke() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tt_rqm_kernels.structuredbench",
            "--suite",
            "qmul",
            "--items",
            "16",
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
    assert report["suite"] == "qmul"
    assert len(report["results"]) == 3
    assert {result["workload"] for result in report["results"]} == {"qmul"}
