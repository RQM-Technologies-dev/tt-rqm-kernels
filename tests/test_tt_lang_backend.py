from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tt_rqm_kernels.backends.tt_lang.availability import (
    TTLangAvailability,
    check_tt_lang_sim,
)
from tt_rqm_kernels.backends.tt_lang.runner import run_qmul_cases
from tt_rqm_kernels.structuredbench import (
    SCHEMA_VERSION,
    BenchmarkCase,
    render_markdown_report,
)


def test_tt_lang_availability_reports_missing_cli() -> None:
    availability = check_tt_lang_sim(sim_cli="/path/that/does/not/exist/tt-lang-sim")

    assert availability.available is False
    assert availability.sim_cli is None
    assert (
        availability.reason
        == "requested tt-lang-sim CLI is not executable: /path/that/does/not/exist/tt-lang-sim"
    )
    assert isinstance(availability.stats_available, bool)


def test_tt_lang_availability_detects_present_stats_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_which(name: str) -> str | None:
        return {
            "tt-lang-sim": "/fake/bin/tt-lang-sim",
            "tt-lang-sim-stats": "/fake/bin/tt-lang-sim-stats",
        }.get(name)

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout="tt-lang-sim 1.2.3\n",
            stderr="",
        )

    monkeypatch.setattr(
        "tt_rqm_kernels.backends.tt_lang.availability.shutil.which",
        fake_which,
    )
    monkeypatch.setattr(
        "tt_rqm_kernels.backends.tt_lang.availability.subprocess.run",
        fake_run,
    )

    availability = check_tt_lang_sim()

    assert availability.available is True
    assert availability.sim_cli == "/fake/bin/tt-lang-sim"
    assert availability.version == "tt-lang-sim 1.2.3"
    assert availability.stats_available is True
    assert availability.stats_cli == "/fake/bin/tt-lang-sim-stats"


def test_tt_lang_availability_detects_missing_stats_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_which(name: str) -> str | None:
        return "/fake/bin/tt-lang-sim" if name == "tt-lang-sim" else None

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout="tt-lang-sim 1.2.3\n",
            stderr="",
        )

    monkeypatch.setattr(
        "tt_rqm_kernels.backends.tt_lang.availability.shutil.which",
        fake_which,
    )
    monkeypatch.setattr(
        "tt_rqm_kernels.backends.tt_lang.availability.subprocess.run",
        fake_run,
    )

    availability = check_tt_lang_sim()

    assert availability.available is True
    assert availability.stats_available is False
    assert availability.stats_cli is None
    assert availability.stats_reason == "tt-lang-sim-stats was not found on PATH."


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
    assert "stats_available" in payload
    assert "stats_cli" in payload
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


def test_tt_lang_runner_does_not_trace_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    commands: list[list[str]] = []
    _mock_tt_lang_runner(monkeypatch, commands, tmp_path)

    case = BenchmarkCase("qmul", 32, 1, 0, "qmul/s")
    report = run_qmul_cases([case], seed=0)

    assert len(commands) == 1
    assert "--trace" not in commands[0]
    assert report["tt_lang_sim"]["trace_enabled"] is False
    assert report["tt_lang_sim"]["stats_available"] is True


def test_tt_lang_runner_traces_when_requested_and_survives_stats_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    commands: list[list[str]] = []
    trace_output = tmp_path / "trace.jsonl"
    stats_output = tmp_path / "stats.txt"
    _mock_tt_lang_runner(
        monkeypatch,
        commands,
        tmp_path,
        stats_returncode=7,
        stats_stdout="partial stats",
        stats_stderr="stats failed",
    )

    case = BenchmarkCase("qmul", 32, 1, 0, "qmul/s")
    report = run_qmul_cases(
        [case],
        seed=0,
        trace_output=trace_output,
        stats_output=stats_output,
    )

    sim_command = commands[0]
    stats_command = commands[1]
    assert "--trace" in sim_command
    assert str(trace_output) in sim_command
    assert stats_command == ["/fake/bin/tt-lang-sim-stats", str(trace_output)]
    assert report["tt_lang_sim"]["trace_enabled"] is True
    assert report["tt_lang_sim"]["trace_path"] == str(trace_output)
    assert report["tt_lang_sim"]["stats_available"] is True
    assert report["tt_lang_sim"]["stats_summary"] is None
    assert "exit code 7" in report["tt_lang_sim"]["stats_error"]
    assert "stats failed" in stats_output.read_text(encoding="utf-8")


def test_tt_lang_markdown_includes_trace_stats_section_without_hardware_claim(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    commands: list[list[str]] = []
    trace_output = tmp_path / "trace.jsonl"
    _mock_tt_lang_runner(
        monkeypatch,
        commands,
        tmp_path,
        stats_stdout="copy_end: 4\ndfb_wait_end: 2\n",
    )

    case = BenchmarkCase("qmul", 32, 1, 0, "qmul/s")
    report = run_qmul_cases([case], seed=0, trace_output=trace_output)
    markdown = render_markdown_report(report)

    assert "## TT-Lang Simulator Trace/Stats" in markdown
    assert "copy_end: 4" in markdown
    assert "not hardware performance" in markdown
    assert "hardware result" not in markdown.lower()


def _mock_tt_lang_runner(
    monkeypatch: pytest.MonkeyPatch,
    commands: list[list[str]],
    tmp_path: Path,
    *,
    stats_returncode: int = 0,
    stats_stdout: str = "copy_end: 4\npipe_send: 2\n",
    stats_stderr: str = "",
) -> None:
    availability = TTLangAvailability(
        available=True,
        sim_cli="/fake/bin/tt-lang-sim",
        version="tt-lang-sim test",
        reason="available",
        stats_cli="/fake/bin/tt-lang-sim-stats",
        stats_available=True,
        stats_reason="available",
    )

    def fake_check_tt_lang_sim(*, sim_cli: str | None = None) -> TTLangAvailability:
        return availability

    def fake_run(
        command: list[str],
        *args: object,
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if command[0].endswith("tt-lang-sim-stats"):
            return subprocess.CompletedProcess(
                args=command,
                returncode=stats_returncode,
                stdout=stats_stdout,
                stderr=stats_stderr,
            )

        if "--trace" in command:
            trace_path = Path(command[command.index("--trace") + 1])
            trace_path.write_text("{\"event\":\"copy_end\"}\n", encoding="utf-8")

        output_path = Path(command[command.index("--json-output") + 1])
        output_path.write_text(
            json.dumps(_fake_tt_lang_report(tmp_path)),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(
        "tt_rqm_kernels.backends.tt_lang.runner.check_tt_lang_sim",
        fake_check_tt_lang_sim,
    )
    monkeypatch.setattr(
        "tt_rqm_kernels.backends.tt_lang.runner.subprocess.run",
        fake_run,
    )


def _fake_tt_lang_report(tmp_path: Path) -> dict[str, object]:
    return {
        "schema": SCHEMA_VERSION,
        "generated_at_utc": "2026-07-03T00:00:00+00:00",
        "suite": "qmul",
        "backend": "tt-lang-sim",
        "device": "functional-simulator",
        "dtype": "float32",
        "seed": 0,
        "simulation": True,
        "tt_lang_sim": {
            "block_items": 32,
            "padded_items": 32,
            "layout": "row-major",
        },
        "results": [
            {
                "suite": "qmul",
                "workload": "qmul",
                "backend": "tt-lang-sim",
                "device": "functional-simulator",
                "dtype": "float32",
                "items": 32,
                "iterations": 1,
                "warmup": 0,
                "structured_shape": "[32, 4]",
                "throughput_unit": "qmul/s",
                "elapsed_s": 0.001,
                "latency_ms": 1.0,
                "throughput": 32000.0,
                "max_abs_error": 0.0,
                "max_rel_error": 0.0,
                "rms_error": 0.0,
                "stability_max_abs": None,
                "scalar_reference_max_abs_error": 0.0,
                "estimated_flops": 896,
                "estimated_flops_per_s": 896000.0,
                "estimated_bytes_read": 1024,
                "estimated_bytes_written": 512,
                "estimated_total_bytes": 1536,
                "effective_gb_per_s": 0.001536,
                "arithmetic_intensity_flops_per_byte": 0.5833333333333334,
                "checksum": 0.0,
            }
        ],
        "torch_version": "test",
        "python_version": "test",
        "platform": str(tmp_path),
    }


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
