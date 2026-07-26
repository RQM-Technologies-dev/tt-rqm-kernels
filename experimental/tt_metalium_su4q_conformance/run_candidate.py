#!/usr/bin/env python3
"""Validate provenance and run the SU4Q N300 conformance candidate."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from tt_rqm_kernels.su4q_conformance import (
    PROTOCOL,
    TT_METAL_COMMIT,
    TT_RQM_COMMIT,
    sha256_file,
)

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[1]
DEFAULT_BINARY = PACKAGE / "build" / "tt_rqm_metalium_su4q_conformance"
SOURCE_SUFFIXES = {".cpp", ".h", ".py", ".txt"}


def source_bundle_sha256() -> str:
    digest = hashlib.sha256()
    for path in sorted(
        candidate
        for candidate in PACKAGE.rglob("*")
        if candidate.is_file()
        and candidate.suffix in SOURCE_SUFFIXES
        and not any(part in {"build", "__pycache__"} for part in candidate.parts)
    ):
        relative = path.relative_to(REPO).as_posix().encode()
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def _git(path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_manifest(path: Path, work_dir: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if payload.get("schema") != PROTOCOL:
        raise ValueError("unsupported SU4Q protocol")
    if payload.get("stage") != "conformance" or payload.get("dtype") != "float32":
        raise ValueError("SU4Q candidate supports FP32 conformance only")
    if payload.get("performance_eligible") is not False:
        raise ValueError("SU4Q session must be performance-ineligible")
    for name, record in payload["inputs"].items():
        input_path = work_dir / record["file"]
        if record["sha256"] != sha256_file(input_path):
            raise ValueError(f"{name} input SHA-256 mismatch")
    return payload


def main() -> int:
    work_value = os.environ.get("TT_RQM_SU4Q_DIR")
    manifest_value = os.environ.get("TT_RQM_SU4Q_MANIFEST")
    metal_value = os.environ.get("TT_METAL_HOME")
    if not work_value or not manifest_value or not metal_value:
        print(
            "TT_RQM_SU4Q_DIR, TT_RQM_SU4Q_MANIFEST, and TT_METAL_HOME are required",
            file=sys.stderr,
        )
        return 2
    try:
        work_dir = Path(work_value).resolve()
        _validate_manifest(Path(manifest_value).resolve(), work_dir)
        metal = Path(metal_value).resolve()
        metal_commit = _git(metal, "rev-parse", "HEAD")
        if metal_commit != TT_METAL_COMMIT:
            raise ValueError(
                f"TT-Metal commit mismatch: expected {TT_METAL_COMMIT}, got {metal_commit}"
            )
        source_commit = _git(REPO, "rev-parse", "HEAD")
        ancestor = subprocess.run(
            ["git", "-C", str(REPO), "merge-base", "--is-ancestor", TT_RQM_COMMIT, source_commit]
        )
        if ancestor.returncode != 0:
            raise ValueError("tt-rqm-kernels source is not based on the pinned commit")
        binary = Path(os.environ.get("TT_RQM_SU4Q_BINARY", DEFAULT_BINARY)).resolve()
        if not binary.is_file():
            raise ValueError(f"SU4Q candidate binary not found: {binary}")
        compiler = subprocess.run(
            [os.environ.get("CXX", "c++"), "--version"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()[0]
        env = os.environ.copy()
        env.update(
            {
                "TT_RQM_SU4Q_CANDIDATE_SHA256": sha256_file(binary),
                "TT_RQM_SU4Q_SOURCE_COMMIT": source_commit,
                "TT_RQM_SU4Q_SOURCE_TREE_CLEAN": str(
                    not bool(_git(REPO, "status", "--porcelain"))
                ).lower(),
                "TT_RQM_SU4Q_SOURCE_BUNDLE_SHA256": source_bundle_sha256(),
                "TT_RQM_SU4Q_BASE_COMMIT": TT_RQM_COMMIT,
                "TT_RQM_SU4Q_TT_METAL_COMMIT": metal_commit,
                "TT_RQM_SU4Q_COMPILER_VERSION": compiler,
                "TT_RQM_SU4Q_RUNTIME_VERSION": f"tt-metal-{TT_METAL_COMMIT}",
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
        print(f"SU4Q candidate preflight failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
