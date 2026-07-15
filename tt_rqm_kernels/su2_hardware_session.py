"""Fail-closed collection for designated SU2ComposeBench hardware sessions."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
from typing import Any, Mapping

import torch

from tt_rqm_kernels.benchmark_integrity import IntegrityError
from tt_rqm_kernels.backends.tenstorrent.su2_compose_persistent import (
    render_su2_markdown,
    run_su2_compose,
)
from tt_rqm_kernels.hardware_session import (
    compare_device_health,
    sha256_file,
    validate_device_health,
)


SESSION_SCHEMA = "tt-rqm-su2-compose-session.v2"


def collect_su2_session(
    *,
    session_dir: Path,
    session_id: str,
    command: str,
    methodology_note: str,
    repository_root: Path,
    tt_metal_root: Path,
    expected_candidate_sha256: str,
    expected_execution_source_commit: str,
    expected_tt_metal_commit: str,
    expected_compiler_version: str | None = None,
    expected_runtime_version: str | None = None,
    execution_source_root: Path | None = None,
    expected_source_tree_sha256: str | None = None,
    stability_preregistration: str | None = None,
    invocation: str | None = None,
    tt_smi_command: str = "tt-smi -s",
    seed: int = 0,
) -> Path:
    """Collect one designated process and retain complete pass or failure evidence."""

    _validate_session_id(session_id)
    repository_root = repository_root.resolve()
    tt_metal_root = tt_metal_root.resolve()
    repo_snapshot = _git_snapshot(repository_root)
    metal_snapshot = _git_snapshot(tt_metal_root)
    session_dir.mkdir(parents=True, exist_ok=False)
    _write(session_dir / "command.txt", (invocation or command) + "\n")

    process_capture: dict[str, str] = {}
    report: dict[str, Any] | None = None
    failure: str | None = None
    pre_raw = ""
    post_health: dict[str, Any] = {}
    candidate_path: Path | None = None
    candidate_hash: str | None = None
    source_trees_clean = not repo_snapshot["status"] and not metal_snapshot["status"]
    source_tree_sha256: str | None = None

    try:
        tokens = shlex.split(command)
        if len(tokens) != 1:
            raise IntegrityError("SU2 collection requires one direct candidate executable")
        candidate_path = Path(tokens[0]).resolve()
        if not candidate_path.is_file() or not os.access(candidate_path, os.X_OK):
            raise IntegrityError(f"candidate is not executable: {candidate_path}")
        candidate_hash = sha256_file(candidate_path)
        _write(
            session_dir / "candidate.sha256",
            f"{candidate_hash}  {candidate_path}\n",
        )
        if candidate_hash != expected_candidate_sha256:
            raise IntegrityError("candidate SHA-256 differs from frozen identity")
        if not source_trees_clean:
            raise IntegrityError("repository or TT-Metal source tree is dirty")
        if metal_snapshot["head"] != expected_tt_metal_commit:
            raise IntegrityError("TT-Metal commit differs from frozen identity")
        if execution_source_root is not None:
            from tt_rqm_kernels.su2_profile import validate_source_tree

            source_tree_sha256 = validate_source_tree(
                collector_root=repository_root,
                source_root=execution_source_root.resolve(),
                source_commit=expected_execution_source_commit,
            )
            if source_tree_sha256 != expected_source_tree_sha256:
                raise IntegrityError("execution source tree differs from frozen identity")

        pre_raw = _run_text(shlex.split(tt_smi_command))
        pre_health = validate_device_health(pre_raw, device_id=0)
        _write(session_dir / "pre-device-health.txt", pre_raw)
        compiler_version = _version_line(["c++", "--version"])
        if expected_compiler_version is not None and compiler_version != expected_compiler_version:
            raise IntegrityError("compiler version differs from frozen identity")
        runtime_version = expected_runtime_version or f"tt-metal-{expected_tt_metal_commit[:8]}"
        environment = {
            "schema": "tt-rqm-su2-compose-environment.v1",
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
                "path": str(candidate_path),
                "sha256": candidate_hash,
                "execution_source_commit": expected_execution_source_commit,
                "execution_source_root": None
                if execution_source_root is None
                else str(execution_source_root.resolve()),
                "source_tree_sha256": source_tree_sha256,
            },
            "device_count": 1,
            "device_id": 0,
            "pre_health": pre_health,
        }
        _write_json(session_dir / "environment.json", environment)

        report = run_su2_compose(
            command=str(candidate_path),
            stage="performance",
            methodology_note=methodology_note,
            seed=seed,
            expected_candidate_sha256=expected_candidate_sha256,
            expected_repository_commit=expected_execution_source_commit,
            process_capture=process_capture,
            candidate_environment={
                "TT_RQM_CHIP_TYPE": "wormhole_b0",
                "TT_RQM_TT_METAL_COMMIT": expected_tt_metal_commit,
                "TT_RQM_COMPILER_VERSION": compiler_version,
                "TT_RQM_RUNTIME_VERSION": runtime_version,
            },
        )
        _write_json(session_dir / "report.json", report)
        _write(session_dir / "report.md", render_su2_markdown(report))
        input_hashes = {
            "schema": "tt-rqm-su2-compose-input-hashes.v1",
            "seed": seed,
            "cases": [
                {
                    "B": result["B"],
                    "K": result["K"],
                    "case_id": result["case_id"],
                    **result["input_hashes"],
                }
                for result in report["results"]
            ],
        }
        _write_json(session_dir / "input-hashes.json", input_hashes)
    except Exception as exc:  # Every designated failure remains an artifact.
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        _write(session_dir / "stdout.txt", process_capture.get("stdout", ""))
        _write(session_dir / "stderr.txt", process_capture.get("stderr", ""))
        try:
            post_raw = _run_text(shlex.split(tt_smi_command))
            _write(session_dir / "post-device-health.txt", post_raw)
            post_health = validate_device_health(post_raw, device_id=0)
            if pre_raw:
                compare_device_health(pre_raw, post_raw, device_id=0)
        except Exception as exc:
            post_health = {"validation_error": f"{type(exc).__name__}: {exc}"}
            failure = failure or str(post_health["validation_error"])

        artifacts = []
        for path in sorted(session_dir.iterdir()):
            if path.is_file() and path.name != "session-manifest.json":
                artifacts.append(
                    {
                        "path": path.name,
                        "role": _artifact_role(path.name),
                        "sha256": sha256_file(path),
                    }
                )
        case_order = [] if report is None else [[r["B"], r["K"]] for r in report["results"]]
        retained_pairs = (
            False
            if report is None
            else all(
                len(result["fused"]["timing_s"]["samples"])
                == len(result["unfused"]["timing_s"]["samples"])
                == 10
                for result in report["results"]
            )
        )
        manifest = {
            "schema": SESSION_SCHEMA,
            "session_id": session_id,
            "collection_status": "passed" if failure is None and report is not None else "failed",
            "failure": failure,
            "designated_stability_session": True,
            "cold_start_host_session": True,
            "no_discarded_performance_runs": True,
            "stable_benchmark": False,
            "benchmark_stage": "performance",
            "device_count": 1,
            "device_id": 0,
            "candidate_sha256": expected_candidate_sha256,
            "execution_source_commit": expected_execution_source_commit,
            "tt_metal_commit": expected_tt_metal_commit,
            "collector_repository_commit": repo_snapshot["head"],
            "source_trees_clean": source_trees_clean,
            "source_tree_sha256": source_tree_sha256,
            "stability_preregistration": stability_preregistration,
            "seed": seed,
            "case_order": case_order,
            "all_expected_paired_samples_retained": retained_pairs,
            "lifecycle": None if report is None else report["lifecycle"],
            "post_health": post_health,
            "artifacts": artifacts,
        }
        _write_json(session_dir / "session-manifest.json", manifest)

    if failure is not None:
        raise IntegrityError(
            f"session {session_id} failed; evidence preserved in {session_dir}: {failure}"
        )
    return session_dir


def _git_snapshot(root: Path) -> dict[str, str]:
    return {
        "path": str(root),
        "head": _run_text(["git", "-C", str(root), "rev-parse", "HEAD"]).strip(),
        "branch": _run_text(["git", "-C", str(root), "rev-parse", "--abbrev-ref", "HEAD"]).strip(),
        "status": _run_text(["git", "-C", str(root), "status", "--porcelain"]),
    }


def _run_text(command: list[str]) -> str:
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout


def _version_line(command: list[str], *, allow_failure: bool = False) -> str:
    try:
        completed = subprocess.run(command, capture_output=True, text=True)
    except FileNotFoundError:
        if allow_failure:
            return "unavailable"
        raise
    if completed.returncode and not allow_failure:
        raise IntegrityError(f"version command failed: {shlex.join(command)}")
    value = (completed.stdout or completed.stderr).strip()
    return value.splitlines()[0] if value else "unavailable"


def _write(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifact_role(name: str) -> str:
    return {
        "candidate.sha256": "candidate-identity",
        "command.txt": "exact-command",
        "environment.json": "environment",
        "input-hashes.json": "input-hashes",
        "post-device-health.txt": "post-device-health",
        "pre-device-health.txt": "pre-device-health",
        "report.json": "hardware-report",
        "report.md": "hardware-report-summary",
        "stderr.txt": "candidate-stderr",
        "stdout.txt": "candidate-stdout",
    }.get(name, "session-evidence")


def _validate_session_id(value: str) -> None:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
    if not value or any(character not in allowed for character in value):
        raise IntegrityError("session ID contains unsupported characters")
