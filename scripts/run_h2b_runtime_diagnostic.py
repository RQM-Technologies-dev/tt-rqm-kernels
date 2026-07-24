#!/usr/bin/env python3
"""Run exactly one development-only H2B D0-D5 runtime-isolation stage."""

from __future__ import annotations

import argparse
from array import array
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import struct
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experimental.tt_metalium_hamiltonian_evolution.run_candidate import source_bundle_sha256
from experimental.tt_metalium_hamiltonian_evolution.check_environment import (
    validate_tt_metal_root,
)
from tt_rqm_kernels.hamiltonian_evolution_runtime_isolation import (
    EVENT_PREFIX,
    EVENT_SCHEMA,
    H2BRuntimeIsolationError,
    InvocationIdentity,
    STAGES,
    build_result,
    build_runner_failure_result,
    parse_event_log,
    render_summary,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def preflight(command: str, output_root: Path) -> tuple[Path, str]:
    command_parts = shlex.split(command)
    if not command_parts:
        raise ValueError("diagnostic command must not be empty")
    executable = Path(command_parts[0]).resolve()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError("diagnostic executable must exist and be executable")
    tt_home_value = os.environ.get("TT_METAL_HOME")
    runtime_root_value = os.environ.get("TT_METAL_RUNTIME_ROOT")
    if not tt_home_value or not runtime_root_value:
        raise ValueError("TT_METAL_HOME and TT_METAL_RUNTIME_ROOT are required")
    tt_home = Path(tt_home_value).resolve()
    runtime_root = Path(runtime_root_value).resolve()
    if tt_home != runtime_root:
        raise ValueError("TT-Metal and runtime roots must resolve to the same checkout")
    tt_commit, _ = validate_tt_metal_root(tt_home)
    if git(ROOT, "status", "--porcelain"):
        raise ValueError("diagnostic source checkout must be clean")
    if git(tt_home, "status", "--porcelain"):
        raise ValueError("TT-Metal checkout must be clean")
    if output_root.exists() and not output_root.is_dir():
        raise ValueError("diagnostic output root is not a directory")
    if any(part in {"benchmarks", "raw", "processed"} for part in output_root.parts):
        raise ValueError("diagnostic output must not use a benchmark evidence path")
    visibility = subprocess.run(
        shlex.split(os.environ.get("TT_RQM_H2B_DEVICE_VISIBILITY_COMMAND", "tt-smi -s")),
        check=False,
        capture_output=True,
        text=True,
    )
    if visibility.returncode or not visibility.stdout.strip():
        raise ValueError("device 0 visibility preflight failed")
    loader = (
        ["otool", "-L", str(executable)] if sys.platform == "darwin" else ["ldd", str(executable)]
    )
    libraries = subprocess.run(loader, check=False, capture_output=True, text=True)
    library_text = libraries.stdout + libraries.stderr
    if libraries.returncode or "not found" in library_text.lower():
        raise ValueError("diagnostic executable shared libraries do not resolve")
    return executable, tt_commit


def prepare_d5_case(work_dir: Path) -> Path:
    coefficients = array("f", [0.0, 0.0, 0.0, 0.0])
    dt = array("f", [0.0])
    coefficients_path = work_dir / "hamiltonians.bin"
    dt_path = work_dir / "dt.bin"
    coefficients_path.write_bytes(coefficients.tobytes())
    dt_path.write_bytes(dt.tobytes())
    manifest = {
        "schema": "tt-rqm-external-hamiltonian-evolution.v1",
        "benchmark": "HamiltonianEvolutionBench",
        "stage": "conformance",
        "dtype": "float32",
        "hamiltonian_shape": [1, 1, 4],
        "dt_shape": [],
        "hbar": 1.0,
        "inputs": {
            "hamiltonians": coefficients_path.name,
            "dt": dt_path.name,
            "hamiltonians_sha256": sha256_file(coefficients_path),
            "dt_sha256": sha256_file(dt_path),
        },
        "outputs": {
            "final_rotors": "final_rotors.bin",
            "final_phases": "final_phases.bin",
            "metrics": "metrics.json",
        },
    }
    manifest_path = work_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _read_float32(path: Path, count: int) -> tuple[float, ...]:
    payload = path.read_bytes()
    if len(payload) != count * 4:
        raise H2BRuntimeIsolationError(f"{path.name} has an invalid byte length")
    return struct.unpack(f"={count}f", payload)


def validate_d5_outputs(work_dir: Path, identity: InvocationIdentity) -> dict[str, object]:
    rotors = _read_float32(work_dir / "final_rotors.bin", 4)
    phases = _read_float32(work_dir / "final_phases.bin", 2)
    values = (*rotors, *phases)
    if not all(math.isfinite(value) for value in values):
        raise H2BRuntimeIsolationError("D5 output contains a nonfinite value")
    expected = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    max_abs_error = max(abs(actual - target) for actual, target in zip(values, expected))
    if max_abs_error > 1e-5:
        raise H2BRuntimeIsolationError("D5 independent identity-oracle validation failed")

    try:
        metrics = json.loads((work_dir / "metrics.json").read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise H2BRuntimeIsolationError("D5 metrics are missing or malformed") from exc
    if not isinstance(metrics, dict):
        raise H2BRuntimeIsolationError("D5 metrics must be a JSON object")
    required_metrics = {
        "schema": "tt-rqm-external-hamiltonian-evolution-metrics.v1",
        "protocol": "tt-rqm-external-hamiltonian-evolution.v1",
        "benchmark": "HamiltonianEvolutionBench",
        "stable_benchmark": False,
        "performance_eligible": False,
        "claim_level": None,
    }
    for key, expected_value in required_metrics.items():
        if metrics.get(key) != expected_value:
            raise H2BRuntimeIsolationError(f"D5 metrics field {key} is invalid")
    metadata = metrics.get("candidate_metadata")
    if not isinstance(metadata, dict):
        raise H2BRuntimeIsolationError("D5 candidate metadata is missing")
    required_metadata = {
        "candidate_sha256": identity.executable_sha256,
        "source_commit": identity.source_commit,
        "source_tree_clean": True,
        "source_bundle_sha256": identity.source_bundle_sha256,
        "tt_metal_commit": identity.tt_metal_commit,
        "runtime_version": identity.runtime_version,
        "device_arch": identity.device_arch,
        "device_count": 1,
        "device_id": identity.device_id,
        "device_create_count": 1,
        "device_close_count": 1,
        "program_count": 2,
        "device_resident_intermediate": True,
        "intermediate_d2h_count": 0,
        "intermediate_h2d_count": 0,
        "host_round_trip_count": 0,
    }
    for key, expected_value in required_metadata.items():
        if metadata.get(key) != expected_value:
            raise H2BRuntimeIsolationError(f"D5 candidate metadata field {key} is invalid")
    if metadata.get("compiler_version") != identity.compiler_version:
        raise H2BRuntimeIsolationError("D5 compiler provenance is invalid")
    return {
        "passed": True,
        "kind": "external_protocol_identity_oracle",
        "max_abs_error": max_abs_error,
        "device_close_completed": True,
        "metrics_schema": metrics["schema"],
    }


def _event(
    diagnostic_id: str,
    name: str,
    *,
    start: float,
    status: str | None = None,
    error: str | None = None,
    validation: dict[str, object] | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "schema": EVENT_SCHEMA,
        "diagnostic_id": diagnostic_id,
        "stage": "d5",
        "event": name,
        "status": status or ("started" if name.endswith("_begin") else "completed"),
        "elapsed_monotonic_s": time.monotonic() - start,
        "process_id": os.getpid(),
        "device_id": 0,
        "error": error,
    }
    if validation is not None:
        record["validation"] = validation
    return record


def _write_result(output_dir: Path, result: dict[str, object]) -> None:
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(render_summary(result), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--command")
    parser.add_argument("--external-command")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("reports/development/h2b-runtime"),
    )
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--acknowledge-development-only", action="store_true")
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    if args.stage in {"d2", "d3", "d4", "d5"} and not args.acknowledge_development_only:
        parser.error("D2-D5 require --acknowledge-development-only")
    if args.stage == "d5":
        if not args.external_command:
            parser.error("D5 requires --external-command")
        if args.command:
            parser.error("D5 does not accept --command")
        selected_command = args.external_command
    else:
        if not args.command:
            parser.error("D0-D4 require --command")
        if args.external_command:
            parser.error("D0-D4 do not accept --external-command")
        selected_command = args.command

    started = datetime.now(timezone.utc)
    monotonic_start = time.monotonic()
    diagnostic_id = (
        f"h2b-runtime-isolation-{args.stage}-{started.strftime('%Y%m%dT%H%M%SZ')}-"
        f"{uuid.uuid4().hex[:8]}"
    )
    output_dir: Path | None = None
    try:
        executable, tt_commit = preflight(selected_command, args.output_root.resolve())
        output_dir = args.output_root.resolve() / diagnostic_id
        output_dir.mkdir(parents=True, exist_ok=False)
        cache = output_dir / "tt-metal-cache"
        cache.mkdir()
        stdout_path = output_dir / "stdout.txt"
        stderr_path = output_dir / "stderr.txt"
        event_path = output_dir / "events.jsonl"
        env = os.environ.copy()
        env["TT_METAL_CACHE"] = str(cache)
        env["TT_RQM_H2B_DIAGNOSTIC_ID"] = diagnostic_id
        env["TT_RQM_H2B_SOURCE_COMMIT"] = git(ROOT, "rev-parse", "HEAD")
        env["TT_RQM_H2B_SOURCE_BUNDLE_SHA256"] = source_bundle_sha256()
        env["TT_RQM_H2B_TT_METAL_COMMIT"] = tt_commit
        env["TT_RQM_H2B_CANDIDATE_SHA256"] = sha256_file(executable)
        env["TT_RQM_H2B_SOURCE_TREE_CLEAN"] = "true"
        env["TT_RQM_H2B_RUNTIME_VERSION"] = f"tt-metal-{tt_commit}"
        compiler = subprocess.run(
            [os.environ.get("CXX", "c++"), "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
        env["TT_RQM_H2B_COMPILER_VERSION"] = compiler.stdout.splitlines()[0]
        if args.stage == "d5":
            manifest = prepare_d5_case(output_dir)
            env["TT_RQM_H2B_DIR"] = str(output_dir)
            env["TT_RQM_H2B_MANIFEST"] = str(manifest)
        identity = InvocationIdentity(
            diagnostic_id,
            args.stage,
            sha256_file(executable),
            env["TT_RQM_H2B_SOURCE_COMMIT"],
            env["TT_RQM_H2B_SOURCE_BUNDLE_SHA256"],
            tt_commit,
            env["TT_RQM_H2B_RUNTIME_VERSION"],
            env["TT_RQM_H2B_COMPILER_VERSION"],
        )
        invocation = shlex.split(selected_command)
        synthetic_events: list[dict[str, object]] = []
        if args.stage == "d5":
            synthetic_events.extend(
                (
                    _event(diagnostic_id, "process_start", start=monotonic_start),
                    _event(diagnostic_id, "external_protocol_begin", start=monotonic_start),
                )
            )
        else:
            invocation.extend(("--stage", args.stage))
        process = subprocess.Popen(
            invocation,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=args.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.terminate()
            try:
                stdout, stderr = process.communicate(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")

        if args.stage == "d5":
            if timed_out or process.returncode != 0:
                error = (
                    f"external protocol timed out after {args.timeout_seconds} seconds"
                    if timed_out
                    else f"external protocol exited with status {process.returncode}"
                )
                synthetic_events.append(
                    _event(
                        diagnostic_id,
                        "failure",
                        start=monotonic_start,
                        status="failed",
                        error=error,
                    )
                )
            else:
                synthetic_events.extend(
                    (
                        _event(
                            diagnostic_id,
                            "external_protocol_end",
                            start=monotonic_start,
                        ),
                        _event(
                            diagnostic_id,
                            "validation_begin",
                            start=monotonic_start,
                        ),
                    )
                )
                try:
                    validation = validate_d5_outputs(output_dir, identity)
                except H2BRuntimeIsolationError as exc:
                    synthetic_events.append(
                        _event(
                            diagnostic_id,
                            "failure",
                            start=monotonic_start,
                            status="failed",
                            error=str(exc),
                        )
                    )
                    process.returncode = 2
                else:
                    synthetic_events.extend(
                        (
                            _event(
                                diagnostic_id,
                                "validation_end",
                                start=monotonic_start,
                                validation=validation,
                            ),
                            _event(
                                diagnostic_id,
                                "process_end",
                                start=monotonic_start,
                            ),
                        )
                    )
            events = synthetic_events
            event_path.write_text(
                "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
                encoding="utf-8",
            )
        else:
            event_lines = [
                line[len(EVENT_PREFIX) :]
                for line in stderr.splitlines()
                if line.startswith(EVENT_PREFIX)
            ]
            event_path.write_text(
                "\n".join(event_lines) + ("\n" if event_lines else ""),
                encoding="utf-8",
            )
            try:
                events = parse_event_log(
                    stderr,
                    stage=args.stage,
                    expected_diagnostic_id=diagnostic_id,
                )
            except H2BRuntimeIsolationError as exc:
                result = build_runner_failure_result(
                    identity,
                    failure_classification="environment"
                    if timed_out and not event_lines
                    else "event_validation",
                    error=str(exc),
                    return_code=process.returncode,
                    stdout_path=Path(stdout_path.name),
                    stderr_path=Path(stderr_path.name),
                    event_log_path=Path(event_path.name),
                    started_at=started.isoformat(),
                    timed_out=timed_out,
                )
                _write_result(output_dir, result)
                print(json.dumps(result, sort_keys=True))
                return 1
        result = build_result(
            identity,
            events,
            return_code=process.returncode,
            stdout_path=Path(stdout_path.name),
            stderr_path=Path(stderr_path.name),
            event_log_path=Path(event_path.name),
            started_at=started.isoformat(),
            timed_out=timed_out,
        )
        _write_result(output_dir, result)
        print(json.dumps(result, sort_keys=True))
        return 0 if result["failure_classification"] is None else 1
    except (
        H2BRuntimeIsolationError,
        OSError,
        ValueError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as exc:
        if output_dir is not None:
            stderr_path = output_dir / "stderr.txt"
            stdout_path = output_dir / "stdout.txt"
            event_path = output_dir / "events.jsonl"
            stderr_path.write_text(str(exc) + "\n", encoding="utf-8")
            stdout_path.touch(exist_ok=True)
            event_path.touch(exist_ok=True)
            identity = InvocationIdentity(
                diagnostic_id,
                args.stage,
                "",
                "",
                "",
                "",
            )
            result = build_runner_failure_result(
                identity,
                failure_classification="runner",
                error=str(exc),
                return_code=None,
                stdout_path=Path(stdout_path.name),
                stderr_path=Path(stderr_path.name),
                event_log_path=Path(event_path.name),
                started_at=started.isoformat(),
            )
            _write_result(output_dir, result)
        print(f"H2B runtime-isolation preflight/runner failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
