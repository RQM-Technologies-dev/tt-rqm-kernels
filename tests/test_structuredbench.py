from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tt_rqm_kernels.structuredbench import (
    SCHEMA_VERSION,
    build_cases,
    render_markdown_report,
    render_table,
    run_suite,
)

FAST_EXTERNAL_QMUL = f"{sys.executable} tests/fixtures/qmul_external_fast.py"


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
    assert report["execution_label"] == "cpu"
    assert report["stable_benchmark"] is False
    assert report["methodology_note"] == (
        "CPU/PyTorch reference run; not a hardware performance result."
    )
    assert len(report["results"]) == 5
    for result in report["results"]:
        assert result["execution_label"] == "cpu"
        assert result["stable_benchmark"] is False
        assert result["methodology_note"] == report["methodology_note"]
        assert result["items"] == 32
        assert result["iterations"] == 1
        assert result["throughput"] > 0.0
        assert result["max_abs_error"] >= 0.0
        assert result["estimated_flops"] > 0
        assert result["estimated_flops_per_s"] > 0.0
        assert result["estimated_bytes_read"] > 0
        assert result["estimated_bytes_written"] > 0
        assert result["estimated_total_bytes"] > 0
        assert result["effective_gb_per_s"] > 0.0
        assert result["arithmetic_intensity_flops_per_byte"] > 0.0


def test_render_table_includes_workloads() -> None:
    report = run_suite("qmul", items_override=16, iterations_override=1, warmup_override=0)
    table = render_table(report)

    assert "StructuredBench" in table
    assert "schema=structuredbench.v1" in table
    assert "qmul" in table


def test_render_markdown_report_includes_hardware_metrics() -> None:
    report = run_suite("qmul", items_override=16, iterations_override=1, warmup_override=0)
    markdown = render_markdown_report(report)

    assert "# StructuredBench Report" in markdown
    assert "## Hardware-Relevant Metrics" in markdown
    assert "estimated_flops_per_s" in markdown


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


def test_module_cli_external_qmul_reference_backend(tmp_path: Path) -> None:
    markdown_output = tmp_path / "external-qmul.md"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tt_rqm_kernels.structuredbench",
            "--suite",
            "qmul",
            "--backend",
            "external-qmul",
            "--external-command",
            FAST_EXTERNAL_QMUL,
            "--items",
            "16",
            "--iters",
            "1",
            "--warmup",
            "0",
            "--execution-label",
            "emulation",
            "--methodology-note",
            "test fixture emulation label",
            "--format",
            "json",
            "--markdown-output",
            str(markdown_output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)

    assert report["schema"] == SCHEMA_VERSION
    assert report["suite"] == "qmul"
    assert report["backend"] == "external-qmul"
    assert report["device"] == "cpu/python-test-fixture"
    assert report["execution_label"] == "emulation"
    assert report["stable_benchmark"] is False
    assert report["methodology_note"] == "test fixture emulation label"
    assert report["protocol"] == "tt-rqm-external-qmul.v1"
    assert len(report["results"]) == 3
    for result in report["results"]:
        assert result["workload"] == "qmul"
        assert result["execution_label"] == "emulation"
        assert result["stable_benchmark"] is False
        assert result["methodology_note"] == "test fixture emulation label"
        assert result["structured_shape"] == "[16, 4]"
        assert result["scalar_reference_max_abs_error"] is not None
        assert result["max_abs_error"] < 1e-4
    markdown = markdown_output.read_text(encoding="utf-8")
    assert "external-qmul candidate harness" in markdown
    assert "This report demonstrates the `external-qmul` candidate protocol" in markdown
    assert "Execution: `emulation`" in markdown
    assert "Stable benchmark: `false`" in markdown
    assert "test fixture emulation label" in markdown
    assert "should not be read as Tenstorrent hardware performance" in markdown


def test_external_qmul_rejects_simulator_execution_label() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tt_rqm_kernels.structuredbench",
            "--suite",
            "qmul",
            "--backend",
            "external-qmul",
            "--external-command",
            FAST_EXTERNAL_QMUL,
            "--execution-label",
            "simulator",
            "--items",
            "16",
            "--iters",
            "1",
            "--warmup",
            "0",
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "external-qmul reports should use cpu, emulation, or hardware" in completed.stderr


def test_external_qmul_rejects_non_qmul_suite() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tt_rqm_kernels.structuredbench",
            "--suite",
            "smoke",
            "--backend",
            "external-qmul",
            "--external-command",
            FAST_EXTERNAL_QMUL,
            "--items",
            "16",
            "--iters",
            "1",
            "--warmup",
            "0",
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "external-qmul backend currently supports --suite qmul only" in completed.stderr


def test_validate_qmul_candidate_script(tmp_path: Path) -> None:
    json_output = tmp_path / "candidate.json"
    markdown_output = tmp_path / "candidate.md"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/validate_qmul_candidate.py",
            "--command",
            FAST_EXTERNAL_QMUL,
            "--items",
            "16",
            "--iters",
            "1",
            "--warmup",
            "0",
            "--execution-label",
            "emulation",
            "--methodology-note",
            "wrapper label pass-through test",
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "schema=structuredbench.v1" in completed.stdout
    report = json.loads(json_output.read_text(encoding="utf-8"))
    assert report["backend"] == "external-qmul"
    assert report["device"] == "cpu/python-test-fixture"
    assert report["execution_label"] == "emulation"
    assert report["stable_benchmark"] is False
    assert report["methodology_note"] == "wrapper label pass-through test"
    assert report["results"][0]["structured_shape"] == "[16, 4]"
    assert report["results"][0]["execution_label"] == "emulation"
    assert report["results"][0]["scalar_reference_max_abs_error"] is not None
    assert "external-qmul candidate harness" in markdown_output.read_text(
        encoding="utf-8"
    )


def test_module_cli_creates_json_and_markdown_outputs(tmp_path: Path) -> None:
    json_output = tmp_path / "nested" / "structuredbench.json"
    markdown_output = tmp_path / "nested" / "structuredbench.md"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "tt_rqm_kernels.structuredbench",
            "--suite",
            "smoke",
            "--items",
            "16",
            "--iters",
            "1",
            "--warmup",
            "0",
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json_output.exists()
    assert markdown_output.exists()
    report = json.loads(json_output.read_text(encoding="utf-8"))
    markdown = markdown_output.read_text(encoding="utf-8")
    assert report["schema"] == SCHEMA_VERSION
    assert "## Hardware-Relevant Metrics" in markdown


def test_tenstorrent_packet_generator(tmp_path: Path) -> None:
    report = run_suite("smoke", items_override=16, iterations_override=1, warmup_override=0)
    input_path = tmp_path / "structuredbench_latest.json"
    output_path = tmp_path / "tenstorrent_packet.md"
    input_path.write_text(json.dumps(report), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/make_tenstorrent_packet.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    packet = output_path.read_text(encoding="utf-8")
    assert "Implemented Stage A TT-Metalium target" in packet
    assert "one Stage A hardware conformance report" in packet
    assert "docs/tenstorrent-landing.md" in packet
    assert "docs/tenstorrent-engineer-copy-paste-packet.md" in packet
    assert "reports/tt_emule_qmul_candidate.md" in packet
    assert "tt-emule evidence" in packet
    assert "QuantumIR for Classical AI Compute" in packet
    assert "AI augmentation" in packet
