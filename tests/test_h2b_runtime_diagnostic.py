from __future__ import annotations

import json
from pathlib import Path
import struct
import subprocess
import sys
from types import SimpleNamespace

import pytest

import scripts.run_h2b_runtime_diagnostic as runner
from scripts.run_h2b_runtime_diagnostic import validate_d5_outputs
from tt_rqm_kernels.hamiltonian_evolution_runtime_isolation import (
    EVENT_PREFIX,
    EVENT_SCHEMA,
    H2BRuntimeIsolationError,
    InvocationIdentity,
    NONCLAIMS,
    STAGE_EVENT_SEQUENCES,
    build_result,
    build_runner_failure_result,
    classify_failure,
    parse_event_log,
    validate_events,
    validate_stage,
)


def event(
    stage: str,
    name: str,
    *,
    elapsed: float = 0.1,
    diagnostic_id: str = "diag",
    process_id: int = 123,
    status: str | None = None,
    **extra: object,
) -> dict:
    return {
        "schema": EVENT_SCHEMA,
        "diagnostic_id": diagnostic_id,
        "stage": stage,
        "event": name,
        "status": status
        or (
            "failed" if name == "failure" else "started" if name.endswith("_begin") else "completed"
        ),
        "elapsed_monotonic_s": elapsed,
        "process_id": process_id,
        "device_id": 0,
        "error": "diagnostic failure" if name == "failure" else None,
        **extra,
    }


def success_events(stage: str, *, validation: dict | None = None) -> list[dict]:
    events: list[dict] = []
    for index, name in enumerate(STAGE_EVENT_SEQUENCES[stage]):
        extra = {}
        if name == "validation_end":
            extra["validation"] = validation or {"passed": True}
        events.append(event(stage, name, elapsed=index / 10, **extra))
    return events


def text(*events: dict) -> str:
    return "\n".join(EVENT_PREFIX + json.dumps(item) for item in events)


def identity(stage: str) -> InvocationIdentity:
    return InvocationIdentity(
        "diag",
        stage,
        "a" * 64,
        "b" * 40,
        "c" * 64,
        "d" * 40,
        "tt-metal-" + "d" * 40,
        "clang version 20",
    )


def paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    return tmp_path / "stdout.txt", tmp_path / "stderr.txt", tmp_path / "events.jsonl"


def test_stage_argument_validation() -> None:
    assert validate_stage("d0") == "d0"
    assert validate_stage("d5") == "d5"
    with pytest.raises(H2BRuntimeIsolationError, match="unsupported"):
        validate_stage("all")


@pytest.mark.parametrize("stage", tuple(STAGE_EVENT_SEQUENCES))
def test_every_complete_stage_sequence_validates(stage: str) -> None:
    events = success_events(
        stage,
        validation={
            "passed": True,
            "device_close_completed": True,
        }
        if stage == "d5"
        else None,
    )
    validate_events(events, stage=stage, expected_diagnostic_id="diag")
    assert (
        parse_event_log(
            text(*events),
            stage=stage,
            expected_diagnostic_id="diag",
        )
        == events
    )


def test_stage_contract_identity_status_duplicate_and_terminal_checks_fail_closed() -> None:
    valid_prefix = [
        event("d0", "process_start", elapsed=0.0),
        event("d0", "device_create_begin", elapsed=0.1),
    ]
    with pytest.raises(H2BRuntimeIsolationError, match="stage contract"):
        validate_events(
            [valid_prefix[0], event("d0", "queue_acquire_begin", elapsed=0.1)],
            stage="d0",
        )
    with pytest.raises(H2BRuntimeIsolationError, match="diagnostic id mismatch"):
        validate_events(
            [valid_prefix[0], event("d0", "device_create_begin", diagnostic_id="other")],
            stage="d0",
        )
    with pytest.raises(H2BRuntimeIsolationError, match="status"):
        validate_events(
            [valid_prefix[0], event("d0", "device_create_begin", status="completed")],
            stage="d0",
        )
    with pytest.raises(H2BRuntimeIsolationError, match="duplicate"):
        validate_events(
            [valid_prefix[0], valid_prefix[1], valid_prefix[1]],
            stage="d0",
        )
    with pytest.raises(H2BRuntimeIsolationError, match="after failure"):
        validate_events(
            [
                valid_prefix[0],
                event("d0", "failure", elapsed=0.1),
                event("d0", "device_create_begin", elapsed=0.2),
            ],
            stage="d0",
        )


def test_event_process_and_monotonic_identity_fail_closed() -> None:
    with pytest.raises(H2BRuntimeIsolationError, match="process id mismatch"):
        validate_events(
            [
                event("d0", "process_start", elapsed=0.0),
                event("d0", "device_create_begin", elapsed=0.1, process_id=999),
            ],
            stage="d0",
        )
    with pytest.raises(H2BRuntimeIsolationError, match="monotonic"):
        validate_events(
            [
                event("d0", "process_start", elapsed=0.2),
                event("d0", "device_create_begin", elapsed=0.1),
            ],
            stage="d0",
        )


def test_successful_d1_loopback_result_is_nonclaiming(tmp_path: Path) -> None:
    stdout, stderr, log = paths(tmp_path)
    result = build_result(
        identity("d1"),
        success_events("d1"),
        return_code=0,
        stdout_path=stdout,
        stderr_path=stderr,
        event_log_path=log,
        started_at="2026-07-17T00:00:00Z",
    )
    assert result["validation"] == {"passed": True}
    assert result["failure_classification"] is None
    assert result["device_close_completed"] is True
    assert result["timed_out"] is False
    assert all(result[key] == value for key, value in NONCLAIMS.items())
    assert result["attempt_count"] == 1 and result["retry_count"] == 0


@pytest.mark.parametrize(
    ("stage", "latest", "expected"),
    (
        ("d0", "device_create_begin", "device_initialization"),
        ("d0", "queue_acquire_begin", "device_initialization"),
        ("d0", "queue_acquire_end", "device_shutdown"),
        ("d1", "h2d_end", "device_to_host"),
        ("d2", "h2d_end", "h2a_execution"),
        ("d2", "execute_h2a_end", "device_to_host"),
        ("d3", "h2d_end", "h1_execution"),
        ("d4", "execute_h2a_end", "h1_execution"),
        ("d5", "external_protocol_begin", "external_protocol"),
        ("d5", "external_protocol_end", "external_protocol"),
        ("d5", "validation_begin", "numerical_validation"),
    ),
)
def test_failure_classification_is_stage_aware(
    stage: str,
    latest: str,
    expected: str,
) -> None:
    sequence = STAGE_EVENT_SEQUENCES[stage]
    prefix = success_events(stage)[: sequence.index(latest) + 1]
    assert classify_failure(stage, prefix) == expected


def test_timeout_result_is_retained_without_retry(tmp_path: Path) -> None:
    events = [
        event("d0", "process_start", elapsed=0.0),
        event("d0", "device_create_begin", elapsed=0.1),
    ]
    stdout, stderr, log = paths(tmp_path)
    result = build_result(
        identity("d0"),
        events,
        return_code=-15,
        stdout_path=stdout,
        stderr_path=stderr,
        event_log_path=log,
        started_at="2026-07-17T00:00:00Z",
        timed_out=True,
    )
    assert result["failure_classification"] == "device_initialization"
    assert result["terminating_signal"] == 15
    assert result["timed_out"] is True
    assert result["retry_count"] == 0


def test_malformed_runner_failure_is_structured_and_nonclaiming(tmp_path: Path) -> None:
    stdout, stderr, log = paths(tmp_path)
    result = build_runner_failure_result(
        identity("d0"),
        failure_classification="event_validation",
        error="malformed event",
        return_code=2,
        stdout_path=stdout,
        stderr_path=stderr,
        event_log_path=log,
        started_at="now",
    )
    assert result["failure_classification"] == "event_validation"
    assert result["runner_error"] == "malformed event"
    assert result["device_close_completed"] is False
    assert all(result[key] == value for key, value in NONCLAIMS.items())


@pytest.mark.parametrize(
    ("stderr", "return_code", "times_out", "expected_classification"),
    (
        (
            text(
                event("d0", "process_start", elapsed=0.0, diagnostic_id="placeholder"),
                event(
                    "d0",
                    "device_create_begin",
                    elapsed=0.1,
                    diagnostic_id="placeholder",
                ),
            ),
            -15,
            True,
            "device_initialization",
        ),
        (EVENT_PREFIX + "{", 2, False, "event_validation"),
    ),
)
def test_runner_retains_timeout_and_malformed_event_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stderr: str,
    return_code: int,
    times_out: bool,
    expected_classification: str,
) -> None:
    executable = tmp_path / "diagnostic"
    executable.write_bytes(b"diagnostic")
    executable.chmod(0o755)
    output_root = tmp_path / "runtime"
    monkeypatch.setattr(
        runner,
        "preflight",
        lambda command, output: (executable, "d" * 40),
    )
    monkeypatch.setattr(runner, "source_bundle_sha256", lambda: "c" * 64)
    monkeypatch.setattr(runner, "git", lambda repo, *args: "b" * 40)
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="clang version 20\n"),
    )

    class FakeProcess:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.returncode = return_code
            self._calls = 0
            self._diagnostic_id = kwargs["env"]["TT_RQM_H2B_DIAGNOSTIC_ID"]

        def communicate(self, timeout: int | None = None) -> tuple[str, str]:
            self._calls += 1
            if times_out and self._calls == 1:
                raise subprocess.TimeoutExpired(["diagnostic"], timeout)
            return "", stderr.replace("placeholder", self._diagnostic_id)

        def terminate(self) -> None:
            self.returncode = -15

        def kill(self) -> None:
            self.returncode = -9

    monkeypatch.setattr(runner.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_h2b_runtime_diagnostic.py",
            "--stage",
            "d0",
            "--command",
            str(executable),
            "--output-root",
            str(output_root),
            "--timeout-seconds",
            "1",
        ],
    )
    assert runner.main() == 1
    result_paths = list(output_root.glob("*/result.json"))
    assert len(result_paths) == 1
    result = json.loads(result_paths[0].read_text(encoding="utf-8"))
    assert result["failure_classification"] == expected_classification
    assert result["timed_out"] is times_out
    assert result["attempt_count"] == 1
    assert result["retry_count"] == 0
    assert all(result[key] == value for key, value in NONCLAIMS.items())


def test_successful_d5_requires_validated_device_close(tmp_path: Path) -> None:
    stdout, stderr, log = paths(tmp_path)
    events = success_events(
        "d5",
        validation={"passed": True, "device_close_completed": True},
    )
    result = build_result(
        identity("d5"),
        events,
        return_code=0,
        stdout_path=stdout,
        stderr_path=stderr,
        event_log_path=log,
        started_at="now",
    )
    assert result["device_close_completed"] is True
    events[-2]["validation"] = {"passed": True, "device_close_completed": False}
    with pytest.raises(H2BRuntimeIsolationError, match="device closure"):
        build_result(
            identity("d5"),
            events,
            return_code=0,
            stdout_path=stdout,
            stderr_path=stderr,
            event_log_path=log,
            started_at="now",
        )


def test_d5_external_outputs_and_provenance_validate(tmp_path: Path) -> None:
    expected_identity = identity("d5")
    (tmp_path / "final_rotors.bin").write_bytes(struct.pack("=4f", 1.0, 0.0, 0.0, 0.0))
    (tmp_path / "final_phases.bin").write_bytes(struct.pack("=2f", 1.0, 0.0))
    metrics = {
        "schema": "tt-rqm-external-hamiltonian-evolution-metrics.v1",
        "protocol": "tt-rqm-external-hamiltonian-evolution.v1",
        "benchmark": "HamiltonianEvolutionBench",
        "stable_benchmark": False,
        "performance_eligible": False,
        "claim_level": None,
        "candidate_metadata": {
            "candidate_sha256": expected_identity.executable_sha256,
            "source_commit": expected_identity.source_commit,
            "source_tree_clean": True,
            "source_bundle_sha256": expected_identity.source_bundle_sha256,
            "tt_metal_commit": expected_identity.tt_metal_commit,
            "runtime_version": expected_identity.runtime_version,
            "compiler_version": expected_identity.compiler_version,
            "device_arch": "wormhole_b0",
            "device_count": 1,
            "device_id": 0,
            "device_create_count": 1,
            "device_close_count": 1,
            "program_count": 2,
            "device_resident_intermediate": True,
            "intermediate_d2h_count": 0,
            "intermediate_h2d_count": 0,
            "host_round_trip_count": 0,
        },
    }
    (tmp_path / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    validation = validate_d5_outputs(tmp_path, expected_identity)
    assert validation["passed"] is True
    assert validation["device_close_completed"] is True
    metrics["candidate_metadata"]["source_commit"] = "wrong"
    (tmp_path / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(H2BRuntimeIsolationError, match="source_commit"):
        validate_d5_outputs(tmp_path, expected_identity)


def test_missing_and_malformed_event_logs() -> None:
    with pytest.raises(H2BRuntimeIsolationError, match="missing"):
        parse_event_log("ordinary stderr", stage="d0")
    with pytest.raises(H2BRuntimeIsolationError, match="malformed"):
        parse_event_log(EVENT_PREFIX + "{", stage="d0")
