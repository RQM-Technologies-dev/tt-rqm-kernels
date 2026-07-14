from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.rqm_tt_quickstart import _default_outputs, _print_next_actions
from scripts.validate_qmul_candidate import (
    EXPECTED_HARDWARE_JSON_OUTPUT,
    EXPECTED_HARDWARE_MARKDOWN_OUTPUT,
    EXPECTED_STAGE_B_CONFORMANCE_JSON_OUTPUT,
    EXPECTED_STAGE_B_CONFORMANCE_MARKDOWN_OUTPUT,
    EXPECTED_STAGE_B_PERFORMANCE_JSON_OUTPUT,
    EXPECTED_STAGE_B_PERFORMANCE_MARKDOWN_OUTPUT,
    _validate_candidate_report,
    _validate_report_args,
)
from tt_rqm_kernels.backends.tenstorrent.availability import (
    DEFAULT_HARDWARE_COMMAND_ENV,
    check_readiness,
    inspect_hardware_command,
    resolve_execution_path,
)
from tt_rqm_kernels.backends.tenstorrent.qmul_external import (
    TenstorrentAdapterError,
    run_external_qmul_inputs,
)
from tt_rqm_kernels.backends.tenstorrent.report import (
    ReportLabelError,
    methodology_note_for_label,
    validate_external_qmul_label,
    validate_stable_benchmark,
)


FAST_EXTERNAL_QMUL = f"{sys.executable} tests/fixtures/qmul_external_fast.py"


def test_tenstorrent_readiness_missing_environment_is_explicit(tmp_path: Path) -> None:
    readiness = check_readiness(
        repo_root=tmp_path,
        report_dir=tmp_path / "reports",
        env={},
    )

    assert readiness.tt_metal_home is None
    assert readiness.tt_emule_home is None
    assert readiness.emule_candidate_script_present is False
    assert readiness.emule_ready is False
    assert readiness.hardware_ready is False
    details = {item.name: item.detail for item in readiness.checks}
    assert "TT_METAL_HOME unset" in details["TT_METAL_HOME"]
    assert "TT_EMULE_HOME unset" in details["TT_EMULE_HOME"]


def test_resolve_hardware_path_requires_configured_command() -> None:
    path = resolve_execution_path("hardware", env={})

    assert path.available is False
    assert path.command is None
    assert path.execution_label == "hardware"
    assert DEFAULT_HARDWARE_COMMAND_ENV in path.reason


def test_hardware_command_preflight_rejects_missing_executable() -> None:
    preflight = inspect_hardware_command("/definitely/missing/rqm_qmul_hw")

    assert preflight.available is False
    assert "executable not found" in preflight.reason


def test_hardware_command_preflight_accepts_safe_fixture_command() -> None:
    preflight = inspect_hardware_command(f"{sys.executable} --version")

    assert preflight.available is True
    assert preflight.executable is not None
    assert "hardware command executable found" in preflight.reason


def test_hardware_command_preflight_rejects_emule_command() -> None:
    preflight = inspect_hardware_command(
        "bash experimental/tt_metalium_qmul/run_candidate_docker.sh"
    )

    assert preflight.available is False
    assert "tt-emule/emulation" in preflight.reason


def test_hardware_command_preflight_rejects_container_command() -> None:
    preflight = inspect_hardware_command("docker run --rm ubuntu true")
    wrapped = inspect_hardware_command("bash -lc 'docker run --rm ubuntu true'")

    assert preflight.available is False
    assert "Docker/container execution" in preflight.reason
    assert wrapped.available is False
    assert "Docker/container execution" in wrapped.reason


def test_report_label_validation_keeps_emulation_out_of_hardware() -> None:
    assert (
        validate_external_qmul_label(
            "emulation",
            command="bash run_candidate_docker.sh",
        )
        == "emulation"
    )
    assert (
        validate_external_qmul_label("hardware", command="/opt/tt/qmul_hw")
        == "hardware"
    )

    with pytest.raises(ReportLabelError, match="tt-emule"):
        validate_external_qmul_label(
            "hardware",
            command="bash experimental/tt_metalium_qmul/run_candidate_docker.sh",
        )
    with pytest.raises(ReportLabelError, match="Docker/container"):
        validate_external_qmul_label("hardware", command="docker run image qmul")
    with pytest.raises(ReportLabelError, match="tt-lang-sim"):
        validate_external_qmul_label("simulator")


def test_report_label_validation_rejects_stable_non_hardware() -> None:
    with pytest.raises(ReportLabelError, match="stable benchmark"):
        validate_stable_benchmark("emulation", stable_benchmark=True)

    note = methodology_note_for_label("hardware", stable_benchmark=False)
    assert "first samples" in note
    assert "stable benchmark" in note


def test_hardware_labeled_report_path_requires_safe_handoff_metadata(
    tmp_path: Path,
) -> None:
    json_output = tmp_path / EXPECTED_HARDWARE_JSON_OUTPUT
    markdown_output = tmp_path / EXPECTED_HARDWARE_MARKDOWN_OUTPUT

    with pytest.raises(ReportLabelError, match="tt-emule"):
        _validate_report_args(
            command="bash experimental/tt_metalium_qmul/run_candidate_docker.sh",
            execution_label="hardware",
            stable_benchmark=False,
            methodology_note="real hardware sample",
            json_output=json_output,
            markdown_output=markdown_output,
        )

    with pytest.raises(ValueError, match="methodology-note"):
        _validate_report_args(
            command="/opt/tt/qmul_hw",
            execution_label="hardware",
            stable_benchmark=False,
            methodology_note=None,
            json_output=json_output,
            markdown_output=markdown_output,
        )

    with pytest.raises(ValueError, match="tt_hardware_qmul_quickstart.json"):
        _validate_report_args(
            command="/opt/tt/qmul_hw",
            execution_label="hardware",
            stable_benchmark=False,
            methodology_note="initial real hardware validation sample",
            json_output=tmp_path / "reports" / "wrong.json",
            markdown_output=markdown_output,
        )

    _validate_report_args(
        command="/opt/tt/qmul_hw",
        execution_label="hardware",
        stable_benchmark=False,
        methodology_note="initial real hardware validation sample",
        json_output=json_output,
        markdown_output=markdown_output,
    )


def test_stage_b_hardware_artifact_names_are_protected(tmp_path: Path) -> None:
    conformance_json = tmp_path / EXPECTED_STAGE_B_CONFORMANCE_JSON_OUTPUT
    conformance_markdown = tmp_path / EXPECTED_STAGE_B_CONFORMANCE_MARKDOWN_OUTPUT
    performance_json = tmp_path / EXPECTED_STAGE_B_PERFORMANCE_JSON_OUTPUT
    performance_markdown = tmp_path / EXPECTED_STAGE_B_PERFORMANCE_MARKDOWN_OUTPUT

    _validate_report_args(
        command="/opt/tt/tt_rqm_metalium_qmul_multicore_candidate",
        execution_label="hardware",
        stable_benchmark=False,
        methodology_note="one Wormhole device 0 multicore conformance",
        json_output=conformance_json,
        markdown_output=conformance_markdown,
        benchmark_stage="conformance",
        candidate="multicore",
    )
    _validate_report_args(
        command="/opt/tt/tt_rqm_metalium_qmul_multicore_candidate",
        execution_label="hardware",
        stable_benchmark=False,
        methodology_note="one Wormhole device 0 first Stage B sample",
        json_output=performance_json,
        markdown_output=performance_markdown,
        benchmark_stage="performance",
        candidate="multicore",
    )

    with pytest.raises(ValueError, match="stage_b_candidate_conformance"):
        _validate_report_args(
            command="/opt/tt/tt_rqm_metalium_qmul_multicore_candidate",
            execution_label="hardware",
            stable_benchmark=False,
            methodology_note="one Wormhole device 0 multicore conformance",
            json_output=tmp_path / EXPECTED_HARDWARE_JSON_OUTPUT,
            markdown_output=conformance_markdown,
            benchmark_stage="conformance",
            candidate="multicore",
        )

    assert _default_outputs(
        "hardware",
        json_output=None,
        markdown_output=None,
        benchmark_stage="conformance",
        candidate="multicore",
    ) == (EXPECTED_STAGE_B_CONFORMANCE_JSON_OUTPUT, EXPECTED_STAGE_B_CONFORMANCE_MARKDOWN_OUTPUT)
    assert _default_outputs(
        "hardware",
        json_output=None,
        markdown_output=None,
        benchmark_stage="performance",
        candidate="multicore",
    ) == (EXPECTED_STAGE_B_PERFORMANCE_JSON_OUTPUT, EXPECTED_STAGE_B_PERFORMANCE_MARKDOWN_OUTPUT)


def test_stage_b_rejects_nonhardware_or_unstaged_validation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="execution-label hardware"):
        _validate_report_args(
            command="/opt/tt/tt_rqm_metalium_qmul_multicore_candidate",
            execution_label="emulation",
            stable_benchmark=False,
            methodology_note="emulation",
            json_output=None,
            markdown_output=None,
            benchmark_stage="conformance",
            candidate="multicore",
        )
    with pytest.raises(ValueError, match="benchmark-stage"):
        _validate_report_args(
            command="/opt/tt/tt_rqm_metalium_qmul_multicore_candidate",
            execution_label="hardware",
            stable_benchmark=False,
            methodology_note="hardware",
            json_output=tmp_path / EXPECTED_STAGE_B_CONFORMANCE_JSON_OUTPUT,
            markdown_output=tmp_path / EXPECTED_STAGE_B_CONFORMANCE_MARKDOWN_OUTPUT,
            benchmark_stage=None,
            candidate="multicore",
        )


def test_candidate_selection_must_match_observed_implementation_class() -> None:
    report = {
        "results": [
            {
                "implementation_class": "scalar_riscv_correctness_baseline",
            }
        ]
    }

    with pytest.raises(ValueError, match="multicore_tensix_sfpu_qmul"):
        _validate_candidate_report("multicore", report)


def test_qmul_external_adapter_requires_command() -> None:
    a = torch.zeros((2, 4), dtype=torch.float32)
    b = torch.zeros((2, 4), dtype=torch.float32)

    with pytest.raises(TenstorrentAdapterError, match="command is not configured"):
        run_external_qmul_inputs(a, b, command=None)


def test_qmul_external_adapter_runs_fixture_command() -> None:
    a = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, -2.0, 3.0, -4.0],
            [-2.0, 0.5, 3.0, -1.0],
            [0.70710677, 0.70710677, 0.0, 0.0],
            [0.9238795, 0.0, 0.0, -0.38268343],
        ],
        dtype=torch.float32,
    )
    b = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [-3.0, 2.0, 1.0, -0.5],
            [1.0, -4.0, 0.5, 2.0],
            [0.9238795, 0.0, 0.38268343, 0.0],
            [0.5, -0.5, 0.5, -0.5],
        ],
        dtype=torch.float32,
    )

    run = run_external_qmul_inputs(a, b, command=FAST_EXTERNAL_QMUL, iterations=1)

    assert run.device == "cpu/python-test-fixture"
    assert run.max_abs_error < 1e-6
    assert run.rms_error < 1e-6
    assert run.throughput > 0.0


def test_quickstart_check_path_is_ci_safe() -> None:
    env = os.environ.copy()
    env.pop(DEFAULT_HARDWARE_COMMAND_ENV, None)
    completed = subprocess.run(
        [sys.executable, "scripts/rqm_tt_quickstart.py", "--check"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert "RQM Tenstorrent qmul quickstart readiness" in completed.stdout
    assert "hardware mode ready:" in completed.stdout
    assert "Next actions" in completed.stdout
    assert "Hardware validation:" in completed.stdout
    assert DEFAULT_HARDWARE_COMMAND_ENV in completed.stdout
    assert "docs/tenstorrent-engineer-copy-paste-packet.md" in completed.stdout
    assert "Do not use tt-emule" in completed.stdout
    assert "--mode hardware" in completed.stdout
    assert "Traceback" not in completed.stderr


def test_quickstart_next_actions_include_emule_command_when_ready(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _print_next_actions(
        SimpleNamespace(
            emule_ready=True,
            hardware_ready=False,
        )
    )

    output = capsys.readouterr().out
    assert "Emulation refresh:" in output
    assert "python scripts/rqm_tt_quickstart.py" in output
    assert "--mode emule" in output
    assert "reports/tt_emule_qmul_candidate.json" in output
    assert "TT_RQM_HARDWARE_QMUL_COMMAND" in output


def test_quickstart_hardware_mode_without_command_fails_cleanly() -> None:
    env = os.environ.copy()
    env.pop(DEFAULT_HARDWARE_COMMAND_ENV, None)
    completed = subprocess.run(
        [sys.executable, "scripts/rqm_tt_quickstart.py", "--mode", "hardware"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 2
    assert "hardware command is not configured" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_quickstart_emule_stable_benchmark_fails_cleanly() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/rqm_tt_quickstart.py",
            "--mode",
            "emule",
            "--stable-benchmark",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "stable benchmark reports are not allowed for --mode emule" in completed.stderr
    assert "Traceback" not in completed.stderr
