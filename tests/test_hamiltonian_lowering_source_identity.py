from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from tt_rqm_kernels.hamiltonian_lowering_source_identity import (
    HamiltonianLoweringSourceIdentityError,
    build_source_manifest,
    source_bundle_sha256,
    source_paths,
    validate_source_manifest,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]


def test_source_inventory_is_deterministic_and_excludes_build_state() -> None:
    first = source_paths(ROOT)
    assert first == source_paths(ROOT)
    assert first
    assert all("build" not in path.parts and "__pycache__" not in path.parts for path in first)
    assert all(not path.is_absolute() for path in first)
    assert source_bundle_sha256(ROOT) == source_bundle_sha256(ROOT)


def test_bundle_hash_changes_with_included_source(tmp_path: Path) -> None:
    _copy_source_roots(tmp_path)
    first = source_bundle_sha256(tmp_path)
    path = tmp_path / source_paths(tmp_path)[0]
    path.write_bytes(path.read_bytes() + b"\n")
    assert source_bundle_sha256(tmp_path) != first


def test_clean_commit_is_required_and_manifest_validates(tmp_path: Path) -> None:
    _copy_source_roots(tmp_path)
    subprocess.run(["git", "init", "-q", tmp_path], check=True)
    subprocess.run(["git", "-C", tmp_path, "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", tmp_path, "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", tmp_path, "add", "."], check=True)
    subprocess.run(["git", "-C", tmp_path, "commit", "-qm", "fixture"], check=True)
    manifest = build_source_manifest(tmp_path)
    manifest_path = tmp_path / "source-manifest.json"
    write_json(manifest_path, manifest)
    result = validate_source_manifest(
        manifest_path, tmp_path, expected_commit=manifest["repository_commit"]
    )
    assert result["source_manifest_valid"] is True
    assert result["file_count"] == len(source_paths(tmp_path))
    assert len(result["source_bundle_sha256"]) == hashlib.sha256().digest_size * 2

    source = tmp_path / source_paths(tmp_path)[0]
    source.write_bytes(source.read_bytes() + b"dirty")
    with pytest.raises(HamiltonianLoweringSourceIdentityError, match="clean committed"):
        build_source_manifest(tmp_path)
    with pytest.raises(HamiltonianLoweringSourceIdentityError, match="inventory mismatch"):
        validate_source_manifest(manifest_path, tmp_path)


def test_manifest_rejects_dirty_tree_declaration(tmp_path: Path) -> None:
    _copy_source_roots(tmp_path)
    subprocess.run(["git", "init", "-q", tmp_path], check=True)
    subprocess.run(["git", "-C", tmp_path, "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", tmp_path, "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", tmp_path, "add", "."], check=True)
    subprocess.run(["git", "-C", tmp_path, "commit", "-qm", "fixture"], check=True)
    payload = build_source_manifest(tmp_path)
    payload["source_tree_clean"] = False
    path = tmp_path / "source-manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(HamiltonianLoweringSourceIdentityError, match="clean tree"):
        validate_source_manifest(path, tmp_path)


def _copy_source_roots(destination: Path) -> None:
    for relative in source_paths(ROOT):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
