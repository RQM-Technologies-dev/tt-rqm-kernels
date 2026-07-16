"""Deterministic source identity for the H2B TT-Metal candidate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

SOURCE_MANIFEST_SCHEMA = "tt-rqm-hamiltonian-evolution-source-manifest.v1"
SOURCE_ROOTS = (
    Path("experimental/tt_metalium_hamiltonian_evolution"),
    Path("experimental/tt_metalium_hamiltonian_lowering_compensated/kernels"),
    Path("experimental/tt_metalium_su2_compose/kernels"),
)
SOURCE_SUFFIXES = frozenset({".cpp", ".h", ".py", ".txt"})
EXCLUDED_PARTS = frozenset({"build", "__pycache__"})


class HamiltonianEvolutionSourceIdentityError(ValueError):
    """Raised when the H2B source identity cannot be reproduced."""


def source_paths(repo_root: Path) -> tuple[Path, ...]:
    paths: list[Path] = []
    for relative_root in SOURCE_ROOTS:
        root = repo_root / relative_root
        if not root.is_dir():
            raise HamiltonianEvolutionSourceIdentityError(f"missing source root: {relative_root}")
        for path in root.rglob("*"):
            relative = path.relative_to(repo_root)
            if (
                path.is_file()
                and path.suffix in SOURCE_SUFFIXES
                and not EXCLUDED_PARTS.intersection(relative.parts)
            ):
                paths.append(relative)
    if not paths:
        raise HamiltonianEvolutionSourceIdentityError("H2B source inventory is empty")
    return tuple(sorted(paths, key=lambda value: value.as_posix()))


def source_bundle_sha256(repo_root: Path) -> str:
    digest = hashlib.sha256()
    for relative in source_paths(repo_root.resolve()):
        name = relative.as_posix().encode()
        payload = (repo_root / relative).read_bytes()
        digest.update(len(name).to_bytes(8, "little"))
        digest.update(name)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def source_scope_clean(repo_root: Path) -> bool:
    paths = [relative.as_posix() for relative in source_paths(repo_root.resolve())]
    modified = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain", "--", *paths],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return not bool(modified)


def build_source_manifest(repo_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    commit = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    entries = [
        {
            "path": relative.as_posix(),
            "size_bytes": (repo_root / relative).stat().st_size,
            "sha256": hashlib.sha256((repo_root / relative).read_bytes()).hexdigest(),
        }
        for relative in source_paths(repo_root)
    ]
    if not source_scope_clean(repo_root):
        raise HamiltonianEvolutionSourceIdentityError(
            "H2B candidate source scope differs from the recorded repository commit"
        )
    return {
        "schema": SOURCE_MANIFEST_SCHEMA,
        "repository_commit": commit,
        "source_scope_clean": True,
        "repository_worktree_clean": not bool(
            subprocess.run(
                ["git", "-C", str(repo_root), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        ),
        "definition": {
            "roots": [path.as_posix() for path in SOURCE_ROOTS],
            "suffixes": sorted(SOURCE_SUFFIXES),
            "excluded_parts": sorted(EXCLUDED_PARTS),
        },
        "source_bundle_sha256": source_bundle_sha256(repo_root),
        "file_count": len(entries),
        "files": entries,
    }


def validate_source_manifest(path: Path, repo_root: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise HamiltonianEvolutionSourceIdentityError("invalid H2B source manifest") from exc
    expected = build_source_manifest(repo_root)
    for key in (
        "schema",
        "repository_commit",
        "source_scope_clean",
        "definition",
        "source_bundle_sha256",
        "file_count",
        "files",
    ):
        if payload.get(key) != expected.get(key):
            raise HamiltonianEvolutionSourceIdentityError(f"source manifest {key} mismatch")
    return payload


def write_source_manifest(path: Path, repo_root: Path) -> dict[str, Any]:
    payload = build_source_manifest(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload
