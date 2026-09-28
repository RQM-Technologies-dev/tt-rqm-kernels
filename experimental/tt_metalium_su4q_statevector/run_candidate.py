#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[1]
PROTOCOL = "tt-rqm-su4q-statevector-conformance.v1"
TT_RQM_BASE = "fffa30784a00656a1a26ee89406633fecc9574ed"
TT_METAL = "9802b80464cfd213b1146189753d8aedf86193fe"

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def git(*args: str, root: Path = REPO) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout.strip()

def source_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in PACKAGE.rglob("*") if candidate.is_file()):
        relative = path.relative_to(REPO).as_posix().encode()
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()

def main() -> int:
    try:
        work = Path(os.environ["TT_RQM_SV_DIR"]).resolve()
        manifest_path = Path(os.environ["TT_RQM_SV_MANIFEST"]).resolve()
        metal = Path(os.environ["TT_METAL_HOME"]).resolve()
        binary = Path(os.environ["TT_RQM_SV_BINARY"]).resolve()
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema") != PROTOCOL or manifest.get("performance_eligible") is not False:
            raise ValueError("unsupported statevector session")
        for record in manifest["inputs"].values():
            path = work / record["file"]
            if sha256(path) != record["sha256"]:
                raise ValueError(f"input hash mismatch: {path.name}")
        if git("rev-parse", "HEAD", root=metal) != TT_METAL:
            raise ValueError("TT-Metal commit mismatch")
        source_commit = git("rev-parse", "HEAD")
        if subprocess.run(
            ["git", "-C", str(REPO), "merge-base", "--is-ancestor", TT_RQM_BASE, source_commit]
        ).returncode:
            raise ValueError("source is not based on the pinned tt-rqm commit")
        if not binary.is_file():
            raise ValueError("candidate binary is missing")
        env = os.environ.copy()
        env.update({
            "TT_RQM_SV_CANDIDATE_SHA256": sha256(binary),
            "TT_RQM_SV_SOURCE_BUNDLE_SHA256": source_hash(),
            "TT_RQM_SV_SOURCE_COMMIT": source_commit,
            "TT_RQM_SV_SOURCE_TREE_CLEAN": str(not bool(git("status", "--porcelain"))).lower(),
            "TT_RQM_SV_TT_METAL_COMMIT": TT_METAL,
        })
        return subprocess.run([str(binary)], env=env, check=False).returncode
    except (KeyError, OSError, ValueError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        print(f"statevector candidate preflight failed: {exc}", file=sys.stderr)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
