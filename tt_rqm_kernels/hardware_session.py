"""Fail-closed collection for isolated Wormhole qmul hardware sessions."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
from typing import Any, Mapping, Sequence

import torch

from tt_rqm_kernels.benchmark_integrity import IntegrityError
from tt_rqm_kernels.backends.tenstorrent.qmul_persistent import (
    render_persistent_markdown,
    run_persistent_qmul,
)


SESSION_SCHEMA = "tt-rqm-benchmark-session.v2"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_qmul_session(
    *,
    session_dir: Path,
    session_id: str,
    command: str,
    benchmark_stage: str,
    methodology_note: str,
    repository_root: Path,
    tt_metal_root: Path,
    expected_candidate_sha256: str,
    expected_execution_source_commit: str,
    expected_tt_metal_commit: str,
    device_id: int = 0,
    output_cb_depth: int = 2,
    seed: int = 0,
    case_specs: Sequence[tuple[int, int, int, int]] | None = None,
    invocation: str | None = None,
    tt_smi_command: str = "tt-smi -s",
    runtime_version: str = "TT-Metalium 0.75.0",
    source_tree_sha256: str | None = None,
) -> Path:
    """Collect one cold-start process and retain complete pass or failure evidence."""

    _validate_session_id(session_id)
    repository_root = repository_root.resolve()
    tt_metal_root = tt_metal_root.resolve()
    session_dir.mkdir(parents=True, exist_ok=False)
    process_capture: dict[str, str] = {}
    artifacts: list[dict[str, str]] = []
    failure: str | None = None
    pre_raw = ""
    post_raw = ""
    report: dict[str, object] | None = None

    try:
        tokens = shlex.split(command)
        if len(tokens) != 1:
            raise IntegrityError("hardware collection requires one direct candidate executable")
        candidate = Path(tokens[0]).resolve()
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            raise IntegrityError(f"candidate is not an executable file: {candidate}")
        observed_candidate_hash = sha256_file(candidate)
        if observed_candidate_hash != expected_candidate_sha256:
            raise IntegrityError("candidate SHA-256 differs from the frozen reference")

        repo_snapshot = _git_snapshot(repository_root)
        metal_snapshot = _git_snapshot(tt_metal_root)
        if repo_snapshot["tracked_status"]:
            raise IntegrityError("tt-rqm-kernels tracked source tree is dirty")
        if metal_snapshot["tracked_status"]:
            raise IntegrityError("tt-metal tracked source tree is dirty")
        if metal_snapshot["head"] != expected_tt_metal_commit:
            raise IntegrityError("tt-metal commit differs from the frozen reference")

        pre_raw = _run_text(shlex.split(tt_smi_command))
        pre_health = validate_device_health(pre_raw, device_id=device_id)
        _write(session_dir / "pre-device-health.txt", pre_raw)
        _write(session_dir / "command.txt", (invocation or command) + "\n")
        _write(session_dir / "candidate.sha256", f"{observed_candidate_hash}  {candidate}\n")

        compiler_version = _version_line(["c++", "--version"])
        environment = {
            "schema": "tt-rqm-hardware-environment.v1",
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "host": platform.node(),
            "platform": platform.platform(),
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "compiler_version": compiler_version,
            "cmake_version": _version_line(["cmake", "--version"]),
            "tt_smi_version": _version_line(["tt-smi", "--version"], allow_failure=True),
            "repository": repo_snapshot,
            "tt_metal": metal_snapshot,
            "candidate": {
                "path": str(candidate),
                "sha256": observed_candidate_hash,
                "execution_source_commit": expected_execution_source_commit,
                "source_tree_sha256": source_tree_sha256,
            },
            "device_id": device_id,
            "visible_device_count": pre_health["visible_device_count"],
            "pre_health": pre_health,
        }
        _write_json(session_dir / "environment.json", environment)

        report = run_persistent_qmul(
            command=str(candidate),
            benchmark_stage=benchmark_stage,
            methodology_note=methodology_note,
            seed=seed,
            device_id=device_id,
            output_cb_depth=output_cb_depth,
            case_specs=case_specs,
            expected_repository_commit=expected_execution_source_commit,
            expected_candidate_sha256=expected_candidate_sha256,
            expected_tt_metal_commit=expected_tt_metal_commit,
            process_capture=process_capture,
            candidate_environment={
                "TT_RQM_CHIP_TYPE": "n300",
                "TT_RQM_TT_METAL_COMMIT": expected_tt_metal_commit,
                "TT_RQM_COMPILER_VERSION": compiler_version,
                "TT_RQM_RUNTIME_VERSION": runtime_version,
            },
            collector_repository_root=repository_root,
        )
        _write_json(session_dir / "report.json", report)
        _write(session_dir / "report.md", render_persistent_markdown(report))
    except Exception as exc:  # Preserve every designated failure before re-raising.
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        _write(session_dir / "stdout.txt", process_capture.get("stdout", ""))
        _write(session_dir / "stderr.txt", process_capture.get("stderr", ""))
        try:
            post_raw = _run_text(shlex.split(tt_smi_command))
            _write(session_dir / "post-device-health.txt", post_raw)
            post_health = validate_device_health(post_raw, device_id=device_id)
            if pre_raw:
                compare_device_health(pre_raw, post_raw, device_id=device_id)
        except Exception as health_exc:
            post_health = {"validation_error": f"{type(health_exc).__name__}: {health_exc}"}
            failure = failure or str(post_health["validation_error"])

        for path in sorted(session_dir.iterdir()):
            if path.is_file() and path.name != "session-manifest.json":
                artifacts.append({"path": path.name, "role": _artifact_role(path.name), "sha256": sha256_file(path)})
        manifest = {
            "schema": SESSION_SCHEMA,
            "session_id": session_id,
            "collection_status": "passed" if failure is None and report is not None else "failed",
            "cold_start_host_session": True,
            "benchmark_stage": benchmark_stage,
            "stable_benchmark": False,
            "device_count": 1,
            "device_id": device_id,
            "output_cb_depth": output_cb_depth,
            "candidate_sha256": expected_candidate_sha256,
            "execution_source_commit": expected_execution_source_commit,
            "source_tree_sha256": source_tree_sha256,
            "tt_metal_commit": expected_tt_metal_commit,
            "seed": seed,
            "case_items": [] if report is None else report["case_items"],
            "lifecycle": None if report is None else report["lifecycle"],
            "post_health": post_health,
            "failure": failure,
            "artifacts": artifacts,
        }
        _write_json(session_dir / "session-manifest.json", manifest)

    if failure is not None:
        raise IntegrityError(f"session {session_id} failed; evidence preserved in {session_dir}: {failure}")
    return session_dir


def validate_device_health(raw: str, *, device_id: int) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
        devices = payload["device_info"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise IntegrityError("tt-smi health output is not valid JSON device data") from exc
    if not isinstance(devices, list) or len(devices) < 2:
        raise IntegrityError("N300 health snapshot must contain both visible device entries")
    if device_id < 0 or device_id >= len(devices):
        raise IntegrityError("selected device is not visible")
    summaries = []
    for index, device in enumerate(devices):
        board = device["board_info"]
        smbus = device["smbus_telem"]
        telemetry = device["telemetry"]
        limits = device["limits"]
        faults = str(smbus.get("FAULTS", "")).lower()
        throttler = str(smbus.get("THROTTLER", "")).lower()
        temperature = float(telemetry["asic_temperature"])
        thermal_limit = float(limits["thm_limit"])
        if board.get("dram_status") is not True:
            raise IntegrityError(f"device {index} DRAM is unhealthy")
        if faults not in {"0x0", "0"}:
            raise IntegrityError(f"device {index} reports hardware faults: {faults}")
        if throttler not in {"0x0", "0"}:
            raise IntegrityError(f"device {index} reports throttling: {throttler}")
        if temperature >= thermal_limit:
            raise IntegrityError(f"device {index} temperature reached its thermal limit")
        summaries.append({
            "device_id": index,
            "board_type": board.get("board_type"),
            "board_id": board.get("board_id"),
            "bus_id": board.get("bus_id"),
            "dram_status": board.get("dram_status"),
            "faults": faults,
            "throttler": throttler,
            "temperature_c": temperature,
            "aiclk_mhz": int(str(telemetry["aiclk"]).strip()),
            "heartbeat": int(str(telemetry["heartbeat"]).strip()),
            "boot_date": smbus.get("BOOT_DATE"),
            "runtime_seconds": smbus.get("RT_SECONDS"),
        })
    return {"visible_device_count": len(devices), "selected_device_id": device_id, "devices": summaries}


def compare_device_health(pre_raw: str, post_raw: str, *, device_id: int) -> None:
    pre = validate_device_health(pre_raw, device_id=device_id)
    post = validate_device_health(post_raw, device_id=device_id)
    if pre["visible_device_count"] != post["visible_device_count"]:
        raise IntegrityError("visible device count changed during session")
    for before, after in zip(pre["devices"], post["devices"], strict=True):
        if before["board_id"] != after["board_id"] or before["boot_date"] != after["boot_date"]:
            raise IntegrityError("device identity or boot state changed during session")
        if before["aiclk_mhz"] != after["aiclk_mhz"]:
            raise IntegrityError("device AICLK changed during session")


def load_session_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") not in {SESSION_SCHEMA, "tt-rqm-benchmark-session.v1"}:
        raise IntegrityError(f"invalid session manifest: {path}")
    return value


def _git_snapshot(root: Path) -> dict[str, Any]:
    return {
        "path": str(root),
        "head": _run_text(["git", "-C", str(root), "rev-parse", "HEAD"]).strip(),
        "branch": _run_text(["git", "-C", str(root), "rev-parse", "--abbrev-ref", "HEAD"]).strip(),
        "tracked_status": _run_text(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"]),
    }


def _run_text(command: list[str]) -> str:
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return completed.stdout


def _version_line(command: list[str], *, allow_failure: bool = False) -> str:
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode and not allow_failure:
        raise IntegrityError(f"version command failed: {shlex.join(command)}")
    text = (completed.stdout or completed.stderr).strip()
    return text.splitlines()[0] if text else "unavailable"


def _write(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifact_role(name: str) -> str:
    return {
        "report.json": "hardware-report",
        "report.md": "hardware-report-summary",
        "environment.json": "environment",
        "pre-device-health.txt": "pre-device-health",
        "post-device-health.txt": "post-device-health",
        "command.txt": "exact-command",
        "candidate.sha256": "candidate-identity",
        "stdout.txt": "candidate-stdout",
        "stderr.txt": "candidate-stderr",
    }.get(name, "session-evidence")


def _validate_session_id(value: str) -> None:
    if not value or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_." for character in value):
        raise IntegrityError("session ID contains unsupported characters")
