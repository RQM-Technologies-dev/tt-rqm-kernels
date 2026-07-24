"""Development-only H2B D0-D5 runtime-isolation result handling."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import json
import re

EVENT_SCHEMA = "tt-rqm-h2b-runtime-isolation-event.v1"
RESULT_SCHEMA = "tt-rqm-h2b-runtime-isolation-result.v1"
EVENT_PREFIX = "H2B_RUNTIME_EVENT "
STAGES = ("d0", "d1", "d2", "d3", "d4", "d5")
VALIDATION_STAGES = frozenset(STAGES[1:])
NONCLAIMS = {
    "development_only": True,
    "benchmark_evidence": False,
    "performance_eligible": False,
    "stable_benchmark": False,
    "claim_level": None,
}

_DEVICE_PREFIX = (
    "process_start",
    "device_create_begin",
    "device_create_end",
    "queue_acquire_begin",
    "queue_acquire_end",
)
_BUFFER_PREFIX = _DEVICE_PREFIX + ("buffer_allocate_begin", "buffer_allocate_end")
_READ_VALIDATE_CLOSE = (
    "d2h_begin",
    "d2h_end",
    "validation_begin",
    "validation_end",
    "device_close_begin",
    "device_close_end",
    "process_end",
)
STAGE_EVENT_SEQUENCES = {
    "d0": _DEVICE_PREFIX + ("device_close_begin", "device_close_end", "process_end"),
    "d1": _BUFFER_PREFIX + ("h2d_begin", "h2d_end") + _READ_VALIDATE_CLOSE,
    "d2": _BUFFER_PREFIX
    + (
        "program_build_h2a_begin",
        "program_build_h2a_end",
        "h2d_begin",
        "h2d_end",
        "execute_h2a_begin",
        "execute_h2a_end",
    )
    + _READ_VALIDATE_CLOSE,
    "d3": _BUFFER_PREFIX
    + (
        "program_build_h1_begin",
        "program_build_h1_end",
        "h2d_begin",
        "h2d_end",
        "execute_h1_begin",
        "execute_h1_end",
    )
    + _READ_VALIDATE_CLOSE,
    "d4": _BUFFER_PREFIX
    + (
        "program_build_h2a_begin",
        "program_build_h2a_end",
        "program_build_h1_begin",
        "program_build_h1_end",
        "h2d_begin",
        "h2d_end",
        "execute_h2a_begin",
        "execute_h2a_end",
        "execute_h1_begin",
        "execute_h1_end",
    )
    + _READ_VALIDATE_CLOSE,
    "d5": (
        "process_start",
        "external_protocol_begin",
        "external_protocol_end",
        "validation_begin",
        "validation_end",
        "process_end",
    ),
}
_KNOWN_EVENTS = frozenset(
    event for sequence in STAGE_EVENT_SEQUENCES.values() for event in sequence
) | {"failure"}


class H2BRuntimeIsolationError(ValueError):
    """Raised when an isolation invocation is malformed or noncompliant."""


@dataclass(frozen=True)
class InvocationIdentity:
    diagnostic_id: str
    stage: str
    executable_sha256: str
    source_commit: str
    source_bundle_sha256: str
    tt_metal_commit: str
    runtime_version: str = ""
    compiler_version: str = ""
    device_id: int = 0
    device_arch: str = "wormhole_b0"


def validate_stage(stage: str) -> str:
    if stage not in STAGES:
        raise H2BRuntimeIsolationError(f"unsupported H2B diagnostic stage: {stage}")
    return stage


def validate_identity(identity: InvocationIdentity) -> None:
    validate_stage(identity.stage)
    if not identity.diagnostic_id:
        raise H2BRuntimeIsolationError("diagnostic id is missing")
    for name, value, length in (
        ("executable sha256", identity.executable_sha256, 64),
        ("source commit", identity.source_commit, 40),
        ("source bundle sha256", identity.source_bundle_sha256, 64),
        ("TT-Metal commit", identity.tt_metal_commit, 40),
    ):
        if not re.fullmatch(rf"[0-9a-f]{{{length}}}", value):
            raise H2BRuntimeIsolationError(f"{name} provenance is invalid")
    if not identity.runtime_version or not identity.compiler_version:
        raise H2BRuntimeIsolationError("runtime/compiler provenance is missing")
    if identity.device_id != 0 or identity.device_arch != "wormhole_b0":
        raise H2BRuntimeIsolationError("device provenance is invalid")


def parse_event_log(
    text: str,
    *,
    stage: str,
    expected_diagnostic_id: str | None = None,
) -> list[dict[str, Any]]:
    validate_stage(stage)
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.startswith(EVENT_PREFIX):
            continue
        try:
            event = json.loads(line[len(EVENT_PREFIX) :])
        except json.JSONDecodeError as exc:
            raise H2BRuntimeIsolationError(
                f"malformed runtime-isolation event at line {line_number}"
            ) from exc
        if not isinstance(event, dict) or event.get("schema") != EVENT_SCHEMA:
            raise H2BRuntimeIsolationError("unsupported runtime-isolation event schema")
        events.append(event)
    if not events:
        raise H2BRuntimeIsolationError("missing runtime-isolation event log")
    validate_events(
        events,
        stage=stage,
        expected_diagnostic_id=expected_diagnostic_id,
    )
    return events


def validate_events(
    events: list[dict[str, Any]],
    *,
    stage: str,
    expected_diagnostic_id: str | None = None,
) -> None:
    validate_stage(stage)
    success_sequence = STAGE_EVENT_SEQUENCES[stage]
    names: list[str] = []
    diagnostic_id: str | None = expected_diagnostic_id
    process_id: int | None = None
    last_elapsed = -1.0

    for event in events:
        if event.get("stage") != stage:
            raise H2BRuntimeIsolationError("reported diagnostic stage mismatch")
        name = event.get("event")
        if name not in _KNOWN_EVENTS:
            raise H2BRuntimeIsolationError("unknown runtime-isolation event")
        if name in names:
            raise H2BRuntimeIsolationError("duplicate runtime-isolation event")
        current_id = event.get("diagnostic_id")
        if not isinstance(current_id, str) or not current_id:
            raise H2BRuntimeIsolationError("event diagnostic id is missing")
        if diagnostic_id is None:
            diagnostic_id = current_id
        elif current_id != diagnostic_id:
            raise H2BRuntimeIsolationError("event diagnostic id mismatch")
        current_pid = event.get("process_id")
        if not isinstance(current_pid, int) or isinstance(current_pid, bool) or current_pid <= 0:
            raise H2BRuntimeIsolationError("event process id is invalid")
        if process_id is None:
            process_id = current_pid
        elif current_pid != process_id:
            raise H2BRuntimeIsolationError("event process id mismatch")
        elapsed = event.get("elapsed_monotonic_s")
        if (
            not isinstance(elapsed, (int, float))
            or isinstance(elapsed, bool)
            or elapsed < 0
            or elapsed < last_elapsed
        ):
            raise H2BRuntimeIsolationError("event monotonic time is invalid")
        last_elapsed = float(elapsed)
        device_id = event.get("device_id")
        if device_id not in (None, 0):
            raise H2BRuntimeIsolationError("event device id is invalid")
        expected_status = (
            "failed" if name == "failure" else "started" if name.endswith("_begin") else "completed"
        )
        if event.get("status") != expected_status:
            raise H2BRuntimeIsolationError("event status is invalid")
        error = event.get("error")
        if name == "failure":
            if not isinstance(error, str) or not error:
                raise H2BRuntimeIsolationError("failure event error is missing")
        elif error is not None:
            raise H2BRuntimeIsolationError("non-failure event contains an error")
        if name == "validation_end":
            validation = event.get("validation")
            if not isinstance(validation, dict) or validation.get("passed") is not True:
                raise H2BRuntimeIsolationError("validation event did not pass")
        names.append(name)

    if names[0] != "process_start":
        raise H2BRuntimeIsolationError("process_start must be the first event")
    if "failure" in names:
        if names[-1] != "failure":
            raise H2BRuntimeIsolationError("lifecycle event appears after failure")
        lifecycle = tuple(names[:-1])
    else:
        lifecycle = tuple(names)
    if lifecycle != success_sequence[: len(lifecycle)]:
        raise H2BRuntimeIsolationError(
            f"runtime-isolation events violate the {stage.upper()} stage contract"
        )


def classify_failure(
    stage: str,
    events: Iterable[dict[str, Any]],
) -> str:
    validate_stage(stage)
    names = [event["event"] for event in events if event["event"] != "failure"]
    latest = names[-1] if names else None
    if latest in (None, "process_start"):
        return "environment"
    if latest in {"external_protocol_begin", "external_protocol_end"}:
        return "external_protocol"
    if latest == "device_create_begin":
        return "device_initialization"
    if latest in {"device_create_end", "queue_acquire_begin"}:
        return "device_initialization"
    if latest == "queue_acquire_end":
        return "device_shutdown" if stage == "d0" else "buffer_allocation"
    if latest == "buffer_allocate_begin":
        return "buffer_allocation"
    if latest in {"buffer_allocate_end", "h2d_begin"}:
        return "host_to_device"
    if latest in {
        "program_build_h2a_begin",
        "program_build_h2a_end",
        "program_build_h1_begin",
        "program_build_h1_end",
    }:
        return "program_build"
    if latest == "h2d_end":
        if stage == "d1":
            return "device_to_host"
        return "h1_execution" if stage == "d3" else "h2a_execution"
    if latest == "execute_h2a_begin":
        return "h2a_execution"
    if latest == "execute_h2a_end":
        return "device_to_host" if stage == "d2" else "h1_execution"
    if latest == "execute_h1_begin":
        return "h1_execution"
    if latest in {"execute_h1_end", "d2h_begin"}:
        return "device_to_host"
    if latest in {"d2h_end", "validation_begin", "external_protocol_end"}:
        return "numerical_validation"
    if latest in {"validation_end", "device_close_begin"}:
        return "device_shutdown"
    return "unknown"


def _result_base(
    identity: InvocationIdentity,
    *,
    stdout_path: Path,
    stderr_path: Path,
    event_log_path: Path,
    started_at: str,
    completed_at: str | None,
) -> dict[str, Any]:
    return {
        "schema": RESULT_SCHEMA,
        "diagnostic_id": identity.diagnostic_id,
        "selected_stage": identity.stage,
        **NONCLAIMS,
        "provenance": {
            "executable_sha256": identity.executable_sha256,
            "source_commit": identity.source_commit,
            "source_bundle_sha256": identity.source_bundle_sha256,
            "tt_metal_commit": identity.tt_metal_commit,
            "runtime_version": identity.runtime_version,
            "compiler_version": identity.compiler_version,
            "device_id": identity.device_id,
            "device_arch": identity.device_arch,
        },
        "started_at": started_at,
        "completed_at": completed_at or datetime.now(timezone.utc).isoformat(),
        "attempt_count": 1,
        "retry_count": 0,
        "paths": {
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "events": str(event_log_path),
        },
    }


def build_result(
    identity: InvocationIdentity,
    events: list[dict[str, Any]],
    *,
    return_code: int,
    stdout_path: Path,
    stderr_path: Path,
    event_log_path: Path,
    started_at: str,
    completed_at: str | None = None,
    timed_out: bool = False,
) -> dict[str, Any]:
    validate_identity(identity)
    validate_events(
        events,
        stage=identity.stage,
        expected_diagnostic_id=identity.diagnostic_id,
    )
    names = [event["event"] for event in events]
    failure = "failure" in names or return_code != 0 or timed_out
    if not failure and tuple(names) != STAGE_EVENT_SEQUENCES[identity.stage]:
        raise H2BRuntimeIsolationError("successful process has an incomplete event sequence")
    if not failure and identity.stage in VALIDATION_STAGES and "validation_end" not in names:
        raise H2BRuntimeIsolationError("successful numerical stage is missing validation_end")
    validation = next(
        (
            event.get("validation")
            for event in reversed(events)
            if event["event"] == "validation_end"
        ),
        None,
    )
    if identity.stage == "d5":
        device_close_completed = bool(
            validation and validation.get("device_close_completed") is True
        )
    else:
        device_close_completed = "device_close_end" in names
    result = {
        **_result_base(
            identity,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            event_log_path=event_log_path,
            started_at=started_at,
            completed_at=completed_at,
        ),
        "last_started_event": next(
            (event["event"] for event in reversed(events) if event.get("status") == "started"),
            None,
        ),
        "last_completed_event": next(
            (event["event"] for event in reversed(events) if event.get("status") == "completed"),
            None,
        ),
        "return_code": return_code if return_code >= 0 else None,
        "terminating_signal": -return_code if return_code < 0 else None,
        "timed_out": timed_out,
        "failure_classification": classify_failure(identity.stage, events) if failure else None,
        "validation": validation,
        "device_close_completed": device_close_completed,
    }
    if not failure and not device_close_completed:
        raise H2BRuntimeIsolationError("successful process did not prove device closure")
    return result


def build_runner_failure_result(
    identity: InvocationIdentity,
    *,
    failure_classification: str,
    error: str,
    return_code: int | None,
    stdout_path: Path,
    stderr_path: Path,
    event_log_path: Path,
    started_at: str,
    timed_out: bool = False,
) -> dict[str, Any]:
    validate_stage(identity.stage)
    if not failure_classification or not error:
        raise H2BRuntimeIsolationError("runner failure details are required")
    return {
        **_result_base(
            identity,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            event_log_path=event_log_path,
            started_at=started_at,
            completed_at=None,
        ),
        "last_started_event": None,
        "last_completed_event": None,
        "return_code": return_code if return_code is not None and return_code >= 0 else None,
        "terminating_signal": -return_code if return_code is not None and return_code < 0 else None,
        "timed_out": timed_out,
        "failure_classification": failure_classification,
        "runner_error": error,
        "validation": None,
        "device_close_completed": False,
    }


def render_summary(result: dict[str, Any]) -> str:
    status = "passed" if result["failure_classification"] is None else "failed"
    return "\n".join(
        (
            "# H2B development runtime-isolation result",
            "",
            f"- Diagnostic ID: `{result['diagnostic_id']}`",
            f"- Stage: `{result['selected_stage']}`",
            f"- Status: `{status}`",
            f"- Failure classification: `{result['failure_classification']}`",
            f"- Last completed event: `{result['last_completed_event']}`",
            f"- Timed out: `{str(result['timed_out']).lower()}`",
            f"- Device close completed: `{str(result['device_close_completed']).lower()}`",
            "- Development only: `true`",
            "- Benchmark evidence: `false`",
            "- Performance eligible: `false`",
            "- Stable benchmark: `false`",
            "- Claim level: `null`",
            "",
        )
    )
