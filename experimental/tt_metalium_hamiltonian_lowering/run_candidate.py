#!/usr/bin/env python3
"""Validate provenance and run the H2A hardware candidate protocol."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

try:
    from .check_environment import PINNED_TT_METAL_COMMIT, validate_tt_metal_root
except ImportError:  # Direct script execution.
    from check_environment import PINNED_TT_METAL_COMMIT, validate_tt_metal_root

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[1]
DEFAULT_BINARY = PACKAGE / "build" / "tt_rqm_metalium_hamiltonian_lowering_candidate"
SOURCE_SUFFIXES = {".cpp", ".h", ".py", ".txt"}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_bundle_sha256(root: Path = PACKAGE) -> str:
    digest = hashlib.sha256()
    for path in sorted(
        candidate
        for candidate in root.rglob("*")
        if candidate.is_file()
        and candidate.suffix in SOURCE_SUFFIXES
        and not any(part in {"build", "__pycache__"} for part in candidate.parts)
    ):
        relative = path.relative_to(root).as_posix().encode()
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _validate_manifest(path: Path, work_dir: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "tt-rqm-external-hamiltonian-lowering.v1":
        raise ValueError("unsupported H2A protocol")
    if payload.get("stage") != "conformance" or payload.get("dtype") != "float32":
        raise ValueError("H2A hardware candidate only supports FP32 conformance")
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("H2A inputs must be an object")
    for name in ("hamiltonians", "dt"):
        input_path = work_dir / str(inputs[name])
        expected = inputs.get(f"{name}_sha256")
        if expected != sha256_file(input_path):
            raise ValueError(f"{name} input SHA-256 mismatch")
    return payload


def main() -> int:
    work = os.environ.get("TT_RQM_H2A_DIR")
    manifest = os.environ.get("TT_RQM_H2A_MANIFEST")
    tt_root_value = os.environ.get("TT_METAL_HOME") or os.environ.get("TT_METALIUM_HOME")
    if not work or not manifest or not tt_root_value:
        print(
            "TT_RQM_H2A_DIR, TT_RQM_H2A_MANIFEST, and TT_METAL_HOME are required", file=sys.stderr
        )
        return 2
    try:
        work_dir = Path(work).resolve()
        _validate_manifest(Path(manifest).resolve(), work_dir)
        tt_root = Path(tt_root_value).resolve()
        tt_commit, _ = validate_tt_metal_root(tt_root)
        binary = Path(os.environ.get("TT_RQM_H2A_BINARY", DEFAULT_BINARY)).resolve()
        if not binary.is_file():
            raise ValueError(f"H2A candidate binary not found: {binary}")
        source_override = os.environ.get("TT_RQM_H2A_SOURCE_COMMIT_OVERRIDE")
        source_commit = source_override or _git("rev-parse", "HEAD")
        source_clean = False if source_override else not bool(_git("status", "--porcelain"))
        compiler = subprocess.run(
            [os.environ.get("CXX", "c++"), "--version"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()[0]
        env = os.environ.copy()
        env.update(
            {
                "TT_RQM_H2A_CANDIDATE_SHA256": sha256_file(binary),
                "TT_RQM_H2A_SOURCE_COMMIT": source_commit,
                "TT_RQM_H2A_SOURCE_TREE_CLEAN": str(source_clean).lower(),
                "TT_RQM_H2A_SOURCE_BUNDLE_SHA256": source_bundle_sha256(),
                "TT_RQM_H2A_TT_METAL_COMMIT": tt_commit,
                "TT_RQM_H2A_COMPILER_VERSION": compiler,
                "TT_RQM_H2A_RUNTIME_VERSION": f"tt-metal-{PINNED_TT_METAL_COMMIT}",
            }
        )
        return subprocess.run([str(binary)], env=env, check=False).returncode
    except (
        KeyError,
        OSError,
        ValueError,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
    ) as exc:
        print(f"H2A candidate preflight failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
