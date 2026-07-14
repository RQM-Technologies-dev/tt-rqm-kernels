from __future__ import annotations

import copy
import json
from pathlib import Path
import re
import subprocess
import sys

import pytest

from tt_rqm_kernels.benchmark_release import (
    BenchmarkReleaseError,
    DEFAULT_MANIFEST,
    generate_release,
    load_manifest,
    sha256_file,
    validate_manifest,
    validate_release,
)

ROOT = Path(__file__).resolve().parents[1]
LEVEL_ONE_MANIFEST = Path("benchmarks/manifests/wormhole-qmul.json")
PROTECTED_HASHES = {
    "reports/tt_hardware_qmul_quickstart.json": "388188443535ef523cdf5c6b9c4f6223ee73fcb546f0900f0d3008a6db98254e",
    "reports/tt_hardware_qmul_quickstart.md": "9d12d5fdd57b2b924feffd04cb803d4714a99a3d17720b492b212b0088e566d5",
    "reports/tt_hardware_qmul_environment.txt": "c2cc05ea8b172d74dcbaa54d95446d6ca2a0a75f2a0415971ce0ad9518b67181",
    "reports/tt_hardware_qmul_stage_b_candidate_conformance.json": "bd025045830d6be7ddc606c0a7dddf5757983b8a6d111499f06b332c74964ea3",
    "reports/tt_hardware_qmul_stage_b_candidate_conformance.md": "a5a8b7804677c687ce3c22f8276568173f47d9059bbd001edf00c49f92e4cf86",
    "reports/tt_hardware_qmul_stage_b_performance.json": "71c2c0554a5d2d0e097002c304b436e49e344cb215a0716334533588c3572bfa",
    "reports/tt_hardware_qmul_stage_b_performance.md": "c99d0e2659cddab3620d98b33f28b174a1b94d03c3429f1084871b6448fc02a9",
    "reports/tt_hardware_qmul_stage_b_architecture_audit.md": "ea66ce5edc73d1392f0e81bbf174dba0e672a583763c42c12d9dd2973d1d658e",
    "reports/tt_hardware_qmul_stage_b_persistent_conformance.json": "abcd0b0f6e9764a8b3ae5b310d2dece01fc5afca991db96eb713563efed26490",
    "reports/tt_hardware_qmul_stage_b_persistent_conformance.md": "5b991c022bfd005520487ea99974186e2a5124c2d07a5bdfe9d48ba08fc04ad3",
    "reports/tt_hardware_qmul_stage_b_persistent_performance.json": "89df35d63d350d12f5339fbecf987149ec5029194848ff04ab0d92766e9da3a0",
    "reports/tt_hardware_qmul_stage_b_persistent_performance.md": "e9dc758a8cdc6ec0846b7e47e3367abaa7f40c02a2a3e991bc22d4a356259423",
    "reports/tt_hardware_qmul_stage_b_persistent_environment.txt": "72d2301ab6d292de6605d41cd78bb431e2dc85541bd2de700ea5da5e73998a73",
    "reports/tt_hardware_qmul_stage_b_persistent_timing_audit.md": "0078ac583ebd97e4780f5dfd739afd230b868f2fc682bb1185f67fdede227bf3",
}


def test_release_manifest_and_canonical_report_validate() -> None:
    manifest = validate_release(ROOT / DEFAULT_MANIFEST, repo_root=ROOT)
    assert manifest["schema"] == "tt-rqm-benchmark-release.v1"
    assert manifest["claim"] == {
        "level": 2,
        "name": "stable_one_device_performance",
        "public_session_count": 3,
        "stable_benchmark": True,
    }


def test_archived_level_one_release_remains_valid() -> None:
    manifest = validate_release(ROOT / LEVEL_ONE_MANIFEST, repo_root=ROOT)
    assert manifest["claim"]["level"] == 1


def test_artifact_hash_tampering_is_rejected() -> None:
    manifest = copy.deepcopy(load_manifest(ROOT / LEVEL_ONE_MANIFEST))
    manifest["artifacts"][0]["sha256"] = "0" * 64
    with pytest.raises(BenchmarkReleaseError, match="SHA-256 mismatch"):
        validate_manifest(manifest, repo_root=ROOT)


def test_claim_level_two_requires_three_independent_sessions() -> None:
    manifest = copy.deepcopy(load_manifest(ROOT / LEVEL_ONE_MANIFEST))
    manifest["claim"].update({"level": 2, "stable_benchmark": True})
    with pytest.raises(BenchmarkReleaseError, match="at least three"):
        validate_manifest(manifest, repo_root=ROOT)


def test_bandwidth_claim_requires_hashed_ceiling_evidence() -> None:
    manifest = copy.deepcopy(load_manifest(ROOT / DEFAULT_MANIFEST))
    manifest["measured_bandwidth_gb_per_s"] = 100.0
    with pytest.raises(BenchmarkReleaseError, match="hardware-ceiling"):
        validate_manifest(manifest, repo_root=ROOT)


def test_generated_summary_and_svgs_are_byte_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    outputs = generate_release(ROOT / DEFAULT_MANIFEST, repo_root=ROOT, destination_root=first)
    assert outputs
    assert outputs == generate_release(
        ROOT / DEFAULT_MANIFEST, repo_root=ROOT, destination_root=second
    )
    for relative in outputs:
        assert (first / relative).read_bytes() == (second / relative).read_bytes()
        assert (ROOT / relative).read_bytes() == (first / relative).read_bytes()
        if relative.suffix == ".svg":
            svg = (first / relative).read_text(encoding="utf-8")
            assert re.search(r'id="[mp][0-9a-f]{10}"', svg) is None
            assert "tt_rqm_" in svg


def test_protected_stage_artifacts_are_unchanged() -> None:
    assert {
        path: sha256_file(ROOT / path) for path in PROTECTED_HASHES
    } == PROTECTED_HASHES


def test_benchmark_documentation_links_resolve() -> None:
    pages = [
        ROOT / "README.md",
        ROOT / "docs/tenstorrent-landing.md",
        *sorted((ROOT / "docs/benchmarks").glob("*.md")),
    ]
    pattern = re.compile(r"!?\[[^]]*\]\(([^)]+)\)")
    missing: list[str] = []
    for page in pages:
        for target in pattern.findall(page.read_text(encoding="utf-8")):
            target = target.split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            if not (page.parent / target).resolve().exists():
                missing.append(f"{page.relative_to(ROOT)} -> {target}")
    assert missing == []


def test_one_command_release_check() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/reproduce_wormhole_qmul.py", "--check"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Claim Level 2" in completed.stdout
    assert "stable_benchmark=true" in completed.stdout


def test_hardware_collection_requires_command_and_isolated_output() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/reproduce_wormhole_qmul.py",
            "--collect-stage",
            "performance",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "--command is required" in completed.stderr


def test_processed_summary_uses_logical_traffic_terminology() -> None:
    payload = json.loads(
        (ROOT / "benchmarks/processed/wormhole-qmul-level2-summary.json").read_text()
    )
    assert payload["claim"]["stable_benchmark"] is True
    assert len(payload["cases"]) == 3
    assert all("logical_traffic_gb_per_s" in case for case in payload["cases"])
    assert "measured_bandwidth_gb_per_s" not in json.dumps(payload)
