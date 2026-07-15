"""Fail-closed validation for retained SU2ComposeBench candidate experiments."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

from tt_rqm_kernels.benchmark_integrity import timing_summary
from tt_rqm_kernels.hardware_session import compare_device_health, validate_device_health


SCHEMA = "tt-rqm-su2-compose-candidate-experiment.v1"
DEFAULT_MANIFEST = Path("benchmarks/manifests/su2-compose-candidate-54b91b-experiment.json")
REPORT_SCHEMA = "tt-rqm-su2-compose-report.v1"
PERFORMANCE_CASES = (
    (32768, 8),
    (8192, 32),
    (2048, 128),
    (512, 512),
    (1024, 128),
    (4096, 128),
    (16384, 128),
    (65536, 128),
)
CONFORMANCE_CASES = ((32, 8), (2048, 8))
_HASH_LINE = re.compile(r"^([0-9a-f]{64})  \./([^\n]+)$")


class SU2CandidateExperimentError(ValueError):
    """Raised when retained candidate-experiment evidence is incomplete or altered."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SU2CandidateExperimentError(f"cannot read candidate manifest {path}: {exc}") from exc
    _require(isinstance(value, dict), "candidate manifest must be an object")
    return value


def validate_candidate_experiment(
    manifest_path: Path = DEFAULT_MANIFEST, *, repo_root: Path | None = None
) -> dict[str, Any]:
    root = (repo_root or Path.cwd()).resolve()
    manifest = load_manifest(_resolve(root, manifest_path))
    validate_manifest(manifest, repo_root=root)
    return manifest


def validate_manifest(manifest: Mapping[str, Any], *, repo_root: Path | None = None) -> None:
    root = (repo_root or Path.cwd()).resolve()
    _require(manifest.get("schema") == SCHEMA, "candidate experiment schema mismatch")
    _require(manifest.get("classification") == "diagnostic_not_release", "classification mismatch")
    claim = _mapping(manifest, "claim")
    _require(claim.get("stable_benchmark") is False, "candidate experiment must remain non-stable")
    _require(claim.get("designated_stability_session") is False, "experiment cannot be designated")

    candidate = _mapping(manifest, "candidate")
    candidate_hash = _hash(candidate, "sha256")
    source_commit = _hash(candidate, "source_commit", length=40)
    tt_metal_commit = _hash(candidate, "tt_metal_commit", length=40)
    compiler_first_line = _text(candidate, "compiler_first_line")
    runtime_version = _text(candidate, "runtime_version")
    collector_commit = _hash(candidate, "collector_commit", length=40)
    _require(candidate.get("device") == "tenstorrent/wormhole-device-0", "device mismatch")

    packages = manifest.get("packages")
    _require(isinstance(packages, list) and len(packages) == 2, "exactly two packages required")
    roles: set[str] = set()
    reports: dict[str, Mapping[str, Any]] = {}
    for package in packages:
        _require(isinstance(package, dict), "package entry must be an object")
        role = _text(package, "role")
        _require(role in {"conformance", "performance"}, f"unsupported package role: {role}")
        _require(role not in roles, f"duplicate package role: {role}")
        roles.add(role)
        directory = _resolve(root, Path(_text(package, "path")))
        _require(directory.is_dir(), f"missing package directory: {directory}")
        inventory = directory / "artifacts.sha256"
        _require(inventory.is_file(), f"missing package inventory: {inventory}")
        _require(
            sha256_file(inventory) == _hash(package, "inventory_sha256"),
            f"inventory SHA-256 mismatch: {directory.name}",
        )
        _validate_inventory(directory, inventory)
        _validate_package_identity(
            directory,
            package,
            candidate_hash=candidate_hash,
            source_commit=source_commit,
            tt_metal_commit=tt_metal_commit,
            compiler_first_line=compiler_first_line,
            collector_commit=collector_commit,
        )
        _validate_health(directory, package)
        report_path = directory / _text(package, "canonical_report")
        _require(report_path.is_file(), f"missing canonical {role} report")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        _validate_report(
            report,
            stage=role,
            candidate_hash=candidate_hash,
            source_commit=source_commit,
            tt_metal_commit=tt_metal_commit,
            compiler_first_line=compiler_first_line,
            runtime_version=runtime_version,
        )
        reports[role] = report

    _require(roles == {"conformance", "performance"}, "package roles are incomplete")
    conformance_identity = _report_identity(reports["conformance"])
    performance_identity = _report_identity(reports["performance"])
    _require(conformance_identity == performance_identity, "canonical report provenance differs")


def _validate_inventory(directory: Path, inventory: Path) -> None:
    entries: dict[str, str] = {}
    for line_number, line in enumerate(inventory.read_text(encoding="utf-8").splitlines(), 1):
        match = _HASH_LINE.fullmatch(line)
        _require(match is not None, f"invalid inventory line {line_number}: {directory.name}")
        expected, relative = match.groups()
        _safe_relative(relative, "inventory path")
        _require(relative != "artifacts.sha256", "inventory must not hash itself")
        _require(relative not in entries, f"duplicate inventory path: {relative}")
        path = directory / relative
        _require(
            path.is_file() and not path.is_symlink(), f"missing inventory artifact: {relative}"
        )
        _require(sha256_file(path) == expected, f"artifact SHA-256 mismatch: {relative}")
        entries[relative] = expected
    observed: set[str] = set()
    for path in directory.rglob("*"):
        _require(not path.is_symlink(), f"package symlink is not allowed: {path}")
        if path.is_file() and path != inventory:
            observed.add(path.relative_to(directory).as_posix())
    _require(set(entries) == observed, f"package inventory coverage mismatch: {directory.name}")


def _validate_package_identity(
    directory: Path,
    package: Mapping[str, Any],
    *,
    candidate_hash: str,
    source_commit: str,
    tt_metal_commit: str,
    compiler_first_line: str,
    collector_commit: str,
) -> None:
    binary = directory / _text(package, "candidate_binary")
    _require(binary.is_file(), f"missing candidate binary: {directory.name}")
    _require(sha256_file(binary) == candidate_hash, f"candidate binary mismatch: {directory.name}")
    candidate_line = (directory / "candidate.sha256").read_text(encoding="utf-8").split()
    _require(candidate_line and candidate_line[0] == candidate_hash, "candidate.sha256 mismatch")
    expected_text = {
        "execution-source-commit.txt": source_commit,
        "tt-metal-commit.txt": tt_metal_commit,
        "collector-repository-commit.txt": collector_commit,
    }
    for name, expected in expected_text.items():
        _require(
            (directory / name).read_text(encoding="utf-8").strip() == expected, f"{name} mismatch"
        )
    compiler_lines = (directory / "compiler-version.txt").read_text(encoding="utf-8").splitlines()
    _require(
        compiler_lines and compiler_lines[0] == compiler_first_line, "compiler version mismatch"
    )
    clean_records = package.get("clean_records")
    _require(isinstance(clean_records, list) and clean_records, "clean-tree records required")
    for relative in clean_records:
        _require(isinstance(relative, str), "clean-tree record must be a path")
        _safe_relative(relative, "clean-tree record")
        path = directory / relative
        _require(path.is_file(), f"missing clean-tree record: {relative}")
        _require(path.read_text(encoding="utf-8") == "", f"dirty tree recorded in {relative}")
    omitted = package.get("documented_missing_records", [])
    _require(isinstance(omitted, list), "documented missing records must be a list")
    for relative in omitted:
        _require(isinstance(relative, str), "documented missing record must be a path")
        _safe_relative(relative, "documented missing record")
        _require(
            not (directory / relative).exists(),
            f"documented missing record unexpectedly exists: {relative}",
        )
    exit_statuses = package.get("exit_statuses")
    _require(isinstance(exit_statuses, dict) and exit_statuses, "exit status contract required")
    for relative, expected in exit_statuses.items():
        _safe_relative(str(relative), "exit status path")
        observed = (directory / str(relative)).read_text(encoding="utf-8").strip()
        _require(observed == str(expected), f"exit status mismatch: {relative}")


def _validate_health(directory: Path, package: Mapping[str, Any]) -> None:
    pairs = package.get("health_pairs")
    _require(isinstance(pairs, list) and pairs, "device-health pairs required")
    for pair in pairs:
        _require(isinstance(pair, dict), "device-health pair must be an object")
        pre_path = directory / _text(pair, "pre")
        post_path = directory / _text(pair, "post")
        _require(pre_path.is_file() and post_path.is_file(), "device-health snapshot missing")
        pre, post = pre_path.read_text(encoding="utf-8"), post_path.read_text(encoding="utf-8")
        validate_device_health(pre, device_id=0)
        validate_device_health(post, device_id=0)
        compare_device_health(pre, post, device_id=0)


def _validate_report(
    report: Mapping[str, Any],
    *,
    stage: str,
    candidate_hash: str,
    source_commit: str,
    tt_metal_commit: str,
    compiler_first_line: str,
    runtime_version: str,
) -> None:
    _require(report.get("schema") == REPORT_SCHEMA, f"{stage} report schema mismatch")
    _require(report.get("benchmark") == "SU2ComposeBench", f"{stage} benchmark mismatch")
    _require(report.get("benchmark_stage") == stage, f"{stage} report stage mismatch")
    _require(report.get("execution_label") == "hardware", f"{stage} report is not hardware")
    _require(report.get("device") == "tenstorrent/wormhole-device-0", f"{stage} device mismatch")
    _require(report.get("performance_eligible") is True, f"{stage} is not performance eligible")
    _require(report.get("stable_benchmark") is False, f"{stage} report must remain non-stable")
    _require(
        report.get("lifecycle")
        == {"close_count": 1, "create_count": 1, "device_count": 1, "device_id": 0},
        f"{stage} lifecycle mismatch",
    )
    provenance = _mapping(report, "provenance")
    candidate = _mapping(provenance, "candidate")
    expected = {
        "candidate_sha256": candidate_hash,
        "build_id": candidate_hash,
        "repository_commit": source_commit,
        "tt_metal_commit": tt_metal_commit,
        "compiler_version": compiler_first_line,
        "runtime_version": runtime_version,
    }
    for key, value in expected.items():
        _require(candidate.get(key) == value, f"{stage} provenance mismatch: {key}")
    _require(
        provenance.get("candidate_sha256") == candidate_hash, "top-level candidate hash mismatch"
    )
    _require(provenance.get("repository_commit") == source_commit, "top-level source mismatch")
    results = report.get("results")
    expected_cases = CONFORMANCE_CASES if stage == "conformance" else PERFORMANCE_CASES
    _require(isinstance(results, list), f"{stage} results must be a list")
    _require(
        [(case.get("B"), case.get("K")) for case in results] == list(expected_cases),
        f"{stage} case order mismatch",
    )
    _finite_tree(report, f"{stage} report")
    for result in results:
        _validate_case(result, stage=stage)


def _validate_case(result: Mapping[str, Any], *, stage: str) -> None:
    batch, steps = int(result["B"]), int(result["K"])
    repeats = 1 if stage == "conformance" else max(1, math.ceil(2_621_440 / (batch * steps)))
    warmups, samples = (0, 1) if stage == "conformance" else (2, 10)
    _require(result.get("repeat_count") == repeats, "case repeat count mismatch")
    _require(result.get("warmup_pairs") == warmups, "case warmup count mismatch")
    _require(result.get("samples") == samples, "case sample count mismatch")
    _require(result.get("stable_benchmark") is False, "case must remain non-stable")
    _require(result.get("performance_eligible") is True, "case is not performance eligible")
    metadata = _mapping(result, "candidate_metadata")
    _require(
        metadata.get("device_count") == 1 and metadata.get("device_id") == 0, "case device mismatch"
    )
    _require(
        metadata.get("core_count") == min(math.ceil(batch / 1024), 56), "case core split mismatch"
    )
    _require(metadata.get("fused_dispatches_per_chain") == 1, "fused dispatch mismatch")
    _require(metadata.get("unfused_dispatches_per_chain") == steps - 1, "unfused dispatch mismatch")
    raw = _mapping(result, "raw_candidate_timings_s")
    expected_order = [
        "fused_first" if index % 2 == 0 else "unfused_first" for index in range(samples)
    ]
    _require(raw.get("paired_order") == expected_order, "paired timing order mismatch")
    for path in ("fused", "unfused"):
        raw_samples = raw.get(f"{path}_samples")
        _require(
            isinstance(raw_samples, list) and len(raw_samples) == samples,
            f"{path} raw sample count mismatch",
        )
        normalized = [float(value) / repeats for value in raw_samples]
        observed = _mapping(_mapping(result, path), "timing_s")
        _require(observed == timing_summary(normalized), f"{path} timing summary mismatch")
        correctness = _mapping(_mapping(result, path), "correctness")
        _require(
            correctness.get("validated_values") == 6 * batch, f"{path} whole-output count mismatch"
        )
        _require(correctness.get("failing_values") == 0, f"{path} correctness failure")
        _require(correctness.get("nonfinite_values") == 0, f"{path} nonfinite result")
        _require(
            float(correctness.get("max_abs_error", math.inf)) <= 1e-4, f"{path} tolerance failure"
        )
    _require(
        float(result.get("cpu_oracle_max_abs_error", math.inf)) <= 1e-11, "CPU oracle mismatch"
    )


def _report_identity(report: Mapping[str, Any]) -> tuple[str, ...]:
    candidate = _mapping(_mapping(report, "provenance"), "candidate")
    return tuple(
        str(candidate[key])
        for key in (
            "candidate_sha256",
            "repository_commit",
            "tt_metal_commit",
            "compiler_version",
            "runtime_version",
        )
    )


def _finite_tree(value: Any, label: str) -> None:
    if isinstance(value, float):
        _require(math.isfinite(value), f"nonfinite value in {label}")
    elif isinstance(value, Mapping):
        for nested in value.values():
            _finite_tree(nested, label)
    elif isinstance(value, list):
        for nested in value:
            _finite_tree(nested, label)


def _resolve(root: Path, relative: Path) -> Path:
    path = relative.resolve() if relative.is_absolute() else (root / relative).resolve()
    _require(path.is_relative_to(root), f"path escapes repository root: {relative}")
    return path


def _safe_relative(value: str, label: str) -> None:
    path = Path(value)
    _require(value != "" and not path.is_absolute(), f"invalid {label}: {value}")
    _require(".." not in path.parts and path.as_posix() == value, f"unsafe {label}: {value}")


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    nested = value.get(key)
    _require(isinstance(nested, dict), f"{key} must be an object")
    return nested


def _text(value: Mapping[str, Any], key: str) -> str:
    observed = value.get(key)
    _require(
        isinstance(observed, str) and observed.strip() == observed and observed, f"invalid {key}"
    )
    return observed


def _hash(value: Mapping[str, Any], key: str, *, length: int = 64) -> str:
    observed = _text(value, key)
    _require(re.fullmatch(f"[0-9a-f]{{{length}}}", observed) is not None, f"invalid {key}")
    return observed


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SU2CandidateExperimentError(message)
