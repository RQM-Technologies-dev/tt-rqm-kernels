"""Collection and deterministic processing for SU2ComposeBench profiler evidence."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from tt_rqm_kernels.hardware_session import compare_device_health, validate_device_health


COLLECTION_SCHEMA = "tt-rqm-su2-compose-profile-session.v1"
PROCESSED_SCHEMA = "tt-rqm-su2-compose-profile-attribution.v1"
EXPECTED_CANDIDATE_SHA256 = "54b91bd921a67bcbda0faaafc2019bbfb931a7f1ef5cef26913d252d0f01da16"
EXPECTED_SOURCE_COMMIT = "3238299a9eea2a44dccd6826a947cac3266dd2f7"
EXPECTED_TT_METAL_COMMIT = "dd2849b5bc6b7a5d38a9eafbeba31ef8d530f8d4"
SOURCE_SUBTREE = "experimental/tt_metalium_su2_compose"
PROFILE_CASES = (
    ("many-trajectories-short-chain", 32768, 8, 32),
    ("balanced", 8192, 32, 8),
    ("one-core-long-chain", 512, 512, 1),
    ("large-batch-56-core", 65536, 128, 56),
)
ROLE_BY_PROCESSOR = {
    "BRISC": "reader",
    "TRISC_0": "compute_unpack",
    "TRISC_1": "compute_math",
    "TRISC_2": "compute_pack",
    "NCRISC": "writer",
}


class SU2ProfileError(ValueError):
    """Raised when a profiler capture is incomplete or internally inconsistent."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_profile_session(
    *,
    candidate: Path,
    collector_root: Path,
    execution_source_root: Path,
    tt_metal_root: Path,
    tracy_capture: Path,
    tracy_csvexport: Path,
    output_root: Path,
    session_id: str,
    expected_candidate_sha256: str = EXPECTED_CANDIDATE_SHA256,
    expected_source_commit: str = EXPECTED_SOURCE_COMMIT,
    expected_tt_metal_commit: str = EXPECTED_TT_METAL_COMMIT,
) -> Path:
    """Capture four exact profiler cases, retaining a fail-closed package."""

    _require(
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", session_id) is not None, "invalid session ID"
    )
    candidate = candidate.resolve()
    collector_root = collector_root.resolve()
    execution_source_root = execution_source_root.resolve()
    tt_metal_root = tt_metal_root.resolve()
    tracy_capture = tracy_capture.resolve()
    tracy_csvexport = tracy_csvexport.resolve()
    session_dir = output_root.resolve() / session_id
    _require(not session_dir.exists(), f"profile session already exists: {session_dir}")
    session_dir.mkdir(parents=True)
    command_path = session_dir / "collection-command.txt"
    command_path.write_text(" ".join(sys.argv) + "\n")
    status = "failed"
    failure: str | None = None
    completed_cases: list[dict[str, Any]] = []
    source_tree_sha256 = ""
    try:
        for path, label in (
            (candidate, "candidate"),
            (collector_root, "collector root"),
            (execution_source_root, "execution source root"),
            (tt_metal_root, "TT-Metal root"),
            (tracy_capture, "tracy-capture"),
            (tracy_csvexport, "tracy-csvexport"),
        ):
            _require(path.exists(), f"missing {label}: {path}")
        _require(candidate.is_file(), "candidate must be a file")
        _require(sha256_file(candidate) == expected_candidate_sha256, "candidate SHA-256 mismatch")
        _require(_git_commit(tt_metal_root) == expected_tt_metal_commit, "TT-Metal commit mismatch")
        _require(_git_status(collector_root) == "", "collector repository is dirty")
        _require(_git_status(tt_metal_root) == "", "TT-Metal repository is dirty")
        source_tree_sha256 = validate_source_tree(
            collector_root=collector_root,
            source_root=execution_source_root,
            source_commit=expected_source_commit,
        )
        environment = {
            "schema": "tt-rqm-su2-compose-profile-environment.v1",
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "host": platform.node(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "candidate": {
                "path": str(candidate),
                "sha256": expected_candidate_sha256,
                "source_commit": expected_source_commit,
                "source_tree_sha256": source_tree_sha256,
            },
            "collector": {"path": str(collector_root), "commit": _git_commit(collector_root)},
            "tt_metal": {"path": str(tt_metal_root), "commit": expected_tt_metal_commit},
            "profiler_environment": {
                "TT_METAL_DEVICE_PROFILER": "1",
                "TT_METAL_PROFILER_MID_RUN_DUMP": "1",
                "TT_METAL_RUNTIME_ROOT": str(tt_metal_root),
                "TRACY_NO_INVARIANT_CHECK": "1",
            },
        }
        _write_json(session_dir / "environment.json", environment)
        for label, batch, steps, cores in PROFILE_CASES:
            completed_cases.append(
                _collect_case(
                    label=label,
                    batch=batch,
                    steps=steps,
                    cores=cores,
                    session_dir=session_dir,
                    candidate=candidate,
                    collector_root=collector_root,
                    execution_source_root=execution_source_root,
                    tt_metal_root=tt_metal_root,
                    tracy_capture=tracy_capture,
                    tracy_csvexport=tracy_csvexport,
                    expected_candidate_sha256=expected_candidate_sha256,
                    expected_source_commit=expected_source_commit,
                    expected_tt_metal_commit=expected_tt_metal_commit,
                    expected_source_tree_sha256=source_tree_sha256,
                )
            )
        processed, markdown = process_profile_session(session_dir)
        _write_json(session_dir / "profile-attribution.json", processed)
        (session_dir / "profile-attribution.md").write_text(markdown)
        status = "passed"
    except Exception as exc:  # Retention is required for every failed diagnostic attempt.
        failure = f"{type(exc).__name__}: {exc}"
    session = {
        "schema": COLLECTION_SCHEMA,
        "session_id": session_id,
        "collection_status": status,
        "failure": failure,
        "classification": "diagnostic_not_stability_evidence",
        "stable_benchmark": False,
        "candidate_sha256": expected_candidate_sha256,
        "source_commit": expected_source_commit,
        "source_tree_sha256": source_tree_sha256,
        "tt_metal_commit": expected_tt_metal_commit,
        "device_id": 0,
        "cases": completed_cases,
        "not_observable": ["circular-buffer wait counters", "SFPU utilization counters"],
        "nonclaims": [
            "no_stability_claim",
            "no_acceleration_claim",
            "no_cpu_speedup_claim",
            "no_measured_bandwidth_claim",
            "no_application_claim",
        ],
    }
    _write_json(session_dir / "session.json", session)
    write_artifact_inventory(session_dir)
    if status != "passed":
        raise SU2ProfileError(failure or "profile collection failed")
    return session_dir


def _collect_case(
    *,
    label: str,
    batch: int,
    steps: int,
    cores: int,
    session_dir: Path,
    candidate: Path,
    collector_root: Path,
    execution_source_root: Path,
    tt_metal_root: Path,
    tracy_capture: Path,
    tracy_csvexport: Path,
    expected_candidate_sha256: str,
    expected_source_commit: str,
    expected_tt_metal_commit: str,
    expected_source_tree_sha256: str,
) -> dict[str, Any]:
    case_id = f"b{batch}-k{steps}-{label}"
    case_dir = session_dir / "cases" / case_id
    profiler_dir = case_dir / "profiler"
    profiler_dir.mkdir(parents=True)
    _require(
        sha256_file(candidate) == expected_candidate_sha256, "candidate changed before capture"
    )
    _require(_git_status(collector_root) == "", "collector repository changed before capture")
    _require(
        _git_commit(tt_metal_root) == expected_tt_metal_commit, "TT-Metal changed before capture"
    )
    _require(_git_status(tt_metal_root) == "", "TT-Metal repository changed before capture")
    _require(
        validate_source_tree(
            collector_root=collector_root,
            source_root=execution_source_root,
            source_commit=expected_source_commit,
        )
        == expected_source_tree_sha256,
        "execution source changed before capture",
    )
    pre = _run(["tt-smi", "-s"])
    (case_dir / "pre-device-health.txt").write_text(pre.stdout)
    _require(pre.returncode == 0, "pre-capture tt-smi failed")
    validate_device_health(pre.stdout, device_id=0)

    generated_logs = tt_metal_root / "generated/profiler/.logs"
    for name in ("profile_log_device.csv", "zone_src_locations.log", "new_zone_src_locations.log"):
        path = generated_logs / name
        if path.exists():
            path.unlink()
    trace_path = profiler_dir / f"{case_id}.tracy"
    capture_command = [str(tracy_capture), "-o", str(trace_path)]
    (case_dir / "tracy-command.txt").write_text(" ".join(capture_command) + "\n")
    capture = subprocess.Popen(
        capture_command,
        cwd=case_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    time.sleep(0.5)
    report_path = case_dir / "report.json"
    runner_command = [
        sys.executable,
        str(collector_root / "scripts/run_su2_compose_profile_case.py"),
        "--command",
        str(candidate),
        "--batch",
        str(batch),
        "--steps",
        str(steps),
        "--expected-candidate-sha256",
        expected_candidate_sha256,
        "--expected-source-commit",
        expected_source_commit,
        "--json-output",
        str(report_path),
        "--markdown-output",
        str(case_dir / "report.md"),
        "--candidate-stdout",
        str(case_dir / "candidate.stdout.txt"),
        "--candidate-stderr",
        str(case_dir / "candidate.stderr.txt"),
    ]
    (case_dir / "runner-command.txt").write_text(" ".join(runner_command) + "\n")
    env = os.environ.copy()
    env.update(
        {
            "TT_METAL_HOME": str(tt_metal_root),
            "TT_METAL_RUNTIME_ROOT": str(tt_metal_root),
            "TT_METAL_DEVICE_PROFILER": "1",
            "TT_METAL_PROFILER_MID_RUN_DUMP": "1",
            "TRACY_NO_INVARIANT_CHECK": "1",
        }
    )
    runner = subprocess.run(
        runner_command,
        cwd=collector_root,
        env=env,
        capture_output=True,
        text=True,
    )
    (case_dir / "runner.stdout.txt").write_text(runner.stdout)
    (case_dir / "runner.stderr.txt").write_text(runner.stderr)
    (case_dir / "runner.exit-status.txt").write_text(f"{runner.returncode}\n")
    try:
        capture_stdout, capture_stderr = capture.communicate(timeout=45)
    except subprocess.TimeoutExpired:
        capture.terminate()
        capture_stdout, capture_stderr = capture.communicate(timeout=10)
    (profiler_dir / "tracy-capture.stdout.txt").write_text(capture_stdout)
    (profiler_dir / "tracy-capture.stderr.txt").write_text(capture_stderr)
    (profiler_dir / "tracy-capture.exit-status.txt").write_text(f"{capture.returncode}\n")
    _require(runner.returncode == 0, f"profile runner failed for {case_id}")
    for name in ("profile_log_device.csv", "zone_src_locations.log", "new_zone_src_locations.log"):
        source = generated_logs / name
        _require(source.is_file(), f"missing device profiler artifact: {name}")
        shutil.copy2(source, profiler_dir / name)
    _require(trace_path.is_file() and trace_path.stat().st_size > 0, "Tracy trace is missing")
    for name, flags in (
        ("tracy-zone-events.csv", ["-u"]),
        ("tracy-zone-statistics.csv", []),
        ("tracy-messages.csv", ["-m"]),
    ):
        exported = _run([str(tracy_csvexport), *flags, str(trace_path)])
        (profiler_dir / name).write_text(exported.stdout)
        (profiler_dir / f"{name}.stderr.txt").write_text(exported.stderr)
        (profiler_dir / f"{name}.exit-status.txt").write_text(f"{exported.returncode}\n")
        _require(exported.returncode == 0, f"Tracy export failed: {name}")
    post = _run(["tt-smi", "-s"])
    (case_dir / "post-device-health.txt").write_text(post.stdout)
    _require(post.returncode == 0, "post-capture tt-smi failed")
    validate_device_health(post.stdout, device_id=0)
    compare_device_health(pre.stdout, post.stdout, device_id=0)
    _require(capture.returncode == 0, f"Tracy capture failed for {case_id}")
    _require(report_path.is_file(), f"profile report missing for {case_id}")
    report = json.loads(report_path.read_text())
    result = report["results"][0]
    _require((result["B"], result["K"]) == (batch, steps), "profile case identity mismatch")
    _require(
        result["warmup_pairs"] == 0 and result["samples"] == 1, "profile timing contract mismatch"
    )
    _require(result["stable_benchmark"] is False, "profile case cannot be stable")
    _require(result["candidate_metadata"]["core_count"] == cores, "profile core count mismatch")
    _require(
        result["raw_candidate_timings_s"]["paired_order"] == ["fused_first"],
        "profile path order mismatch",
    )
    return {"id": case_id, "B": batch, "K": steps, "core_count": cores, "status": "passed"}


def validate_source_tree(*, collector_root: Path, source_root: Path, source_commit: str) -> str:
    """Require an untracked source package to byte-match a committed subtree."""

    prefix = f"{SOURCE_SUBTREE}/"
    listed = _git(
        collector_root,
        ["ls-tree", "-r", "--name-only", source_commit, SOURCE_SUBTREE],
    ).splitlines()
    expected = [path for path in listed if path.startswith(prefix)]
    _require(expected, "source commit does not contain the SU2 candidate subtree")
    observed = sorted(
        path.relative_to(source_root).as_posix()
        for path in source_root.rglob("*")
        if path.is_file()
    )
    relative_expected = sorted(path[len(prefix) :] for path in expected)
    _require(observed == relative_expected, "execution-source file set differs from frozen commit")
    digest = hashlib.sha256()
    for relative in relative_expected:
        committed = subprocess.run(
            ["git", "show", f"{source_commit}:{prefix}{relative}"],
            cwd=collector_root,
            capture_output=True,
        )
        _require(committed.returncode == 0, f"cannot read committed source file: {relative}")
        observed_bytes = (source_root / relative).read_bytes()
        _require(
            observed_bytes == committed.stdout, f"execution-source content differs: {relative}"
        )
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(observed_bytes).digest())
    return digest.hexdigest()


def parse_device_profile(path: Path, *, steps: int, core_count: int) -> dict[str, Any]:
    """Convert KERNEL zones into fused/unfused per-role dispatch attribution."""

    with path.open(encoding="utf-8") as handle:
        handle.readline()
        header = [value.strip() for value in handle.readline().split(",")]
        rows = list(csv.DictReader(handle, fieldnames=header))
    opened: dict[tuple[Any, ...], int] = {}
    records: list[dict[str, Any]] = []
    observed_zone_names: set[str] = set()
    for row in rows:
        zone = row["zone name"].strip()
        observed_zone_names.add(zone)
        if not zone.endswith("-KERNEL"):
            continue
        processor = row["RISC processor type"].strip()
        if processor not in ROLE_BY_PROCESSOR:
            continue
        core = (int(row["core_x"]), int(row["core_y"]))
        key = (core, processor, row["timer_id"].strip(), zone)
        timestamp = int(row["time[cycles since reset]"])
        if row["type"].strip() == "ZONE_START":
            _require(key not in opened, "overlapping device profiler zone")
            opened[key] = timestamp
        elif row["type"].strip() == "ZONE_END":
            _require(key in opened, "device profiler zone end without start")
            start = opened.pop(key)
            records.append(
                {
                    "core": core,
                    "processor": processor,
                    "role": ROLE_BY_PROCESSOR[processor],
                    "start": start,
                    "end": timestamp,
                    "cycles": timestamp - start,
                }
            )
    _require(not opened, "unterminated device profiler zones")
    brisc = sorted(
        (record for record in records if record["processor"] == "BRISC"),
        key=lambda value: value["start"],
    )
    _require(len(brisc) == steps * core_count, "device profiler dispatch count mismatch")
    starts: list[int] = []
    for offset in range(0, len(brisc), core_count):
        chunk = brisc[offset : offset + core_count]
        _require(
            len({record["core"] for record in chunk}) == core_count, "incomplete BRISC dispatch"
        )
        starts.append(min(record["start"] for record in chunk))
    dispatches: list[dict[str, Any]] = []
    for index, start in enumerate(starts):
        lower = -math.inf if index == 0 else (starts[index - 1] + start) / 2
        upper = math.inf if index + 1 == len(starts) else (start + starts[index + 1]) / 2
        selected = [record for record in records if lower <= record["start"] < upper]
        roles: dict[str, Any] = {}
        intervals: list[tuple[int, int]] = []
        for processor, role in ROLE_BY_PROCESSOR.items():
            values = [record for record in selected if record["processor"] == processor]
            _require(len(values) == core_count, f"incomplete {processor} dispatch")
            cycles = [int(record["cycles"]) for record in values]
            role_start, role_end = (
                min(record["start"] for record in values),
                max(record["end"] for record in values),
            )
            intervals.append((role_start, role_end))
            roles[role] = {
                "count": len(cycles),
                "min_cycles": min(cycles),
                "median_cycles": statistics.median(cycles),
                "max_cycles": max(cycles),
            }
        overlap = max(0, min(end for _, end in intervals) - max(begin for begin, _ in intervals))
        dispatches.append(
            {
                "index": index,
                "path": "fused" if index == 0 else "unfused",
                "role_overlap_cycles": overlap,
                "roles": roles,
            }
        )
    return {
        "dispatch_count": len(dispatches),
        "expected_dispatch_count": steps,
        "fused_dispatch_count": 1,
        "unfused_dispatch_count": steps - 1,
        "paths": {
            "fused": _summarize_dispatches(dispatches[:1]),
            "unfused": _summarize_dispatches(dispatches[1:]),
        },
        "circular_buffer_waits": (
            "observed"
            if any("CB-" in name and "WAIT" in name for name in observed_zone_names)
            else "not_observable"
        ),
    }


def _summarize_dispatches(dispatches: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require(dispatches, "profile path contains no dispatches")
    roles: dict[str, Any] = {}
    for role in ROLE_BY_PROCESSOR.values():
        medians = [float(dispatch["roles"][role]["median_cycles"]) for dispatch in dispatches]
        maxima = [float(dispatch["roles"][role]["max_cycles"]) for dispatch in dispatches]
        roles[role] = {
            "median_of_core_medians_cycles": statistics.median(medians),
            "median_of_core_maxima_cycles": statistics.median(maxima),
            "maximum_cycles": max(maxima),
        }
    collapsed = {
        "reader": roles["reader"]["median_of_core_maxima_cycles"],
        "compute": max(
            roles["compute_unpack"]["median_of_core_maxima_cycles"],
            roles["compute_math"]["median_of_core_maxima_cycles"],
            roles["compute_pack"]["median_of_core_maxima_cycles"],
        ),
        "writer": roles["writer"]["median_of_core_maxima_cycles"],
    }
    critical = max(collapsed, key=collapsed.get)
    return {
        "dispatch_count": len(dispatches),
        "roles": roles,
        "critical_device_role": critical,
        "all_role_overlap_dispatch_fraction": sum(
            1 for dispatch in dispatches if int(dispatch["role_overlap_cycles"]) > 0
        )
        / len(dispatches),
    }


def parse_tracy_statistics(path: Path, *, timed_pair_s: float) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected: dict[str, dict[str, Any]] = {}
    for name in (
        "EnqueueProgram",
        "FDMeshCommandQueue::finish",
        "FDMeshCommandQueue::finish_nolock",
    ):
        matching = [row for row in rows if row.get("name") == name]
        if matching:
            selected[name] = {
                "total_ns": sum(int(row["total_ns"]) for row in matching),
                "count": sum(int(row["counts"]) for row in matching),
            }
    direct = sum(
        value["total_ns"]
        for key, value in selected.items()
        if key in {"EnqueueProgram", "FDMeshCommandQueue::finish"}
    )
    return {
        "zones": selected,
        "timed_pair_s": timed_pair_s,
        "direct_dispatch_and_finish_fraction_of_timed_pair": direct / (timed_pair_s * 1e9),
    }


def process_profile_session(session_dir: Path) -> tuple[dict[str, Any], str]:
    cases: list[dict[str, Any]] = []
    for label, batch, steps, cores in PROFILE_CASES:
        case_id = f"b{batch}-k{steps}-{label}"
        case_dir = session_dir / "cases" / case_id
        report = json.loads((case_dir / "report.json").read_text())
        result = report["results"][0]
        stdout = (case_dir / "candidate.stdout.txt").read_text()
        _require(
            "markers were dropped" not in stdout.lower(), f"dropped profiler markers: {case_id}"
        )
        device = parse_device_profile(
            case_dir / "profiler/profile_log_device.csv", steps=steps, core_count=cores
        )
        raw = result["raw_candidate_timings_s"]
        timed_pair_s = float(raw["fused_samples"][0]) + float(raw["unfused_samples"][0])
        host = parse_tracy_statistics(
            case_dir / "profiler/tracy-zone-statistics.csv", timed_pair_s=timed_pair_s
        )
        cases.append(
            {
                "id": case_id,
                "B": batch,
                "K": steps,
                "core_count": cores,
                "device": device,
                "host": host,
                "fused_s": float(result["fused"]["timing_s"]["median"]),
                "unfused_s": float(result["unfused"]["timing_s"]["median"]),
                "stable_benchmark": False,
            }
        )
    fused_roles = [case["device"]["paths"]["fused"]["critical_device_role"] for case in cases]
    primary = max(set(fused_roles), key=lambda role: (fused_roles.count(role), role))
    payload = {
        "schema": PROCESSED_SCHEMA,
        "classification": "diagnostic_not_stability_evidence",
        "stable_benchmark": False,
        "candidate_sha256": EXPECTED_CANDIDATE_SHA256,
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "tt_metal_commit": EXPECTED_TT_METAL_COMMIT,
        "cases": cases,
        "fused_primary_device_role": primary,
        "circular_buffer_waits": "not_observable_with_pinned_profiler",
        "architecture_decision": "pending_review",
        "nonclaims": [
            "no_stability_claim",
            "no_acceleration_claim",
            "no_cpu_speedup_claim",
            "no_measured_bandwidth_claim",
            "no_application_claim",
        ],
    }
    lines = [
        "# SU2ComposeBench profiler attribution",
        "",
        "Diagnostic Device Program Profiler and Tracy evidence for the exact retained `54b91b…` candidate. Individual captures and this aggregate remain `stable_benchmark=false`.",
        "",
        "| case | cores | fused ms | unfused ms | fused device role | unfused device role | dispatch + finish / timed pair |",
        "|---|---:|---:|---:|---|---|---:|",
    ]
    for case in cases:
        paths = case["device"]["paths"]
        fraction = case["host"]["direct_dispatch_and_finish_fraction_of_timed_pair"]
        lines.append(
            f"| B={case['B']:,}, K={case['K']} | {case['core_count']} | {case['fused_s'] * 1e3:.3f} | "
            f"{case['unfused_s'] * 1e3:.3f} | {paths['fused']['critical_device_role']} | "
            f"{paths['unfused']['critical_device_role']} | {fraction:.3f} |"
        )
    lines.extend(
        [
            "",
            f"The most frequent fused critical device role is `{primary}`. Reader, compute, and writer KERNEL scopes are reported separately in the machine-readable artifact.",
            "",
            "The pinned profiler does not expose direct circular-buffer wait or SFPU-utilization counters. Their absence is not interpreted as zero wait or full utilization.",
            "",
            "No stability, acceleration, CPU-speedup, measured-bandwidth, or application claim is made.",
            "",
        ]
    )
    return payload, "\n".join(lines)


def write_artifact_inventory(directory: Path) -> None:
    inventory = directory / "artifacts.sha256"
    lines = []
    for path in sorted(
        value for value in directory.rglob("*") if value.is_file() and value != inventory
    ):
        relative = path.relative_to(directory).as_posix()
        lines.append(f"{sha256_file(path)}  ./{relative}")
    inventory.write_text("\n".join(lines) + "\n")


def _git(root: Path, args: Sequence[str]) -> str:
    completed = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
    _require(completed.returncode == 0, f"git {' '.join(args)} failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def _git_commit(root: Path) -> str:
    return _git(root, ["rev-parse", "HEAD"])


def _git_status(root: Path) -> str:
    return _git(root, ["status", "--short", "--untracked-files=no"])


def _run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SU2ProfileError(message)
