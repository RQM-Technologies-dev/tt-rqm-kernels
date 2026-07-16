"""Deterministic clean-source identity for the compensated H2A candidate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

SOURCE_MANIFEST_SCHEMA = "tt-rqm-h2a-source-manifest.v1"
SOURCE_ROOTS = (
    Path("experimental/tt_metalium_hamiltonian_lowering"),
    Path("experimental/tt_metalium_hamiltonian_lowering_compensated"),
)
ADDITIONAL_SOURCE_FILES = (
    Path("tt_rqm_kernels/hamiltonian_lowering_benchmark.py"),
    Path("tt_rqm_kernels/hamiltonian_lowering_candidate.py"),
    Path("tt_rqm_kernels/hamiltonian_lowering_source_identity.py"),
)
SOURCE_SUFFIXES = frozenset({".cpp", ".h", ".py", ".txt"})
EXCLUDED_PARTS = frozenset({"build", "__pycache__"})


class HamiltonianLoweringSourceIdentityError(ValueError):
    """Raised when an H2A source identity cannot be reproduced exactly."""


def git_output(repo_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def source_paths(repo_root: Path) -> tuple[Path, ...]:
    """Return the exact deterministic candidate-source file list."""

    repo_root = repo_root.resolve()
    paths: list[Path] = []
    for relative_root in SOURCE_ROOTS:
        root = repo_root / relative_root
        if not root.is_dir():
            raise HamiltonianLoweringSourceIdentityError(f"missing source root: {relative_root}")
        for path in root.rglob("*"):
            if (
                path.is_file()
                and path.suffix in SOURCE_SUFFIXES
                and not EXCLUDED_PARTS.intersection(path.relative_to(repo_root).parts)
            ):
                paths.append(path.relative_to(repo_root))
    for relative in ADDITIONAL_SOURCE_FILES:
        if not (repo_root / relative).is_file():
            raise HamiltonianLoweringSourceIdentityError(f"missing source file: {relative}")
        paths.append(relative)
    if not paths:
        raise HamiltonianLoweringSourceIdentityError("H2A source file list is empty")
    return tuple(sorted(paths, key=lambda value: value.as_posix()))


def source_entries(repo_root: Path) -> list[dict[str, Any]]:
    repo_root = repo_root.resolve()
    return [
        {
            "path": relative.as_posix(),
            "sha256": hashlib.sha256((repo_root / relative).read_bytes()).hexdigest(),
            "size_bytes": (repo_root / relative).stat().st_size,
        }
        for relative in source_paths(repo_root)
    ]


def source_bundle_sha256(repo_root: Path) -> str:
    """Hash relative names and bytes without timestamps or absolute paths."""

    repo_root = repo_root.resolve()
    digest = hashlib.sha256()
    for relative in source_paths(repo_root):
        name = relative.as_posix().encode()
        payload = (repo_root / relative).read_bytes()
        digest.update(len(name).to_bytes(8, "little"))
        digest.update(name)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def build_source_manifest(repo_root: Path, *, require_clean: bool = True) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    commit = git_output(repo_root, "rev-parse", "HEAD")
    dirty = git_output(repo_root, "status", "--porcelain", "--untracked-files=all")
    if require_clean and dirty:
        raise HamiltonianLoweringSourceIdentityError(
            "designated H2A source identity requires a clean committed repository"
        )
    return {
        "schema": SOURCE_MANIFEST_SCHEMA,
        "repository_commit": commit,
        "source_tree_clean": not bool(dirty),
        "definition": {
            "roots": [path.as_posix() for path in SOURCE_ROOTS],
            "additional_files": [path.as_posix() for path in ADDITIONAL_SOURCE_FILES],
            "included_suffixes": sorted(SOURCE_SUFFIXES),
            "excluded_path_parts": sorted(EXCLUDED_PARTS),
            "hash_encoding": "uint64_le(name_length) || name || uint64_le(payload_length) || payload",
            "absolute_paths_included": False,
            "timestamps_included": False,
        },
        "source_bundle_sha256": source_bundle_sha256(repo_root),
        "files": source_entries(repo_root),
    }


def validate_source_manifest(
    manifest_path: Path, repo_root: Path, *, expected_commit: str | None = None
) -> dict[str, Any]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise HamiltonianLoweringSourceIdentityError("invalid or missing source manifest") from exc
    if manifest.get("schema") != SOURCE_MANIFEST_SCHEMA:
        raise HamiltonianLoweringSourceIdentityError("source manifest schema mismatch")
    commit = manifest.get("repository_commit")
    if expected_commit is not None and commit != expected_commit:
        raise HamiltonianLoweringSourceIdentityError("source manifest commit mismatch")
    if manifest.get("source_tree_clean") is not True:
        raise HamiltonianLoweringSourceIdentityError("source manifest must bind a clean tree")
    expected_files = source_entries(repo_root)
    if manifest.get("files") != expected_files:
        raise HamiltonianLoweringSourceIdentityError("source manifest file inventory mismatch")
    bundle = source_bundle_sha256(repo_root)
    if manifest.get("source_bundle_sha256") != bundle:
        raise HamiltonianLoweringSourceIdentityError("source bundle SHA-256 mismatch")
    return {
        "source_manifest_valid": True,
        "repository_commit": commit,
        "source_bundle_sha256": bundle,
        "file_count": len(expected_files),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
