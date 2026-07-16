"""Cross-check machine-readable releases against current human status surfaces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tt_rqm_kernels.benchmark_release import validate_release as validate_qmul_release
from tt_rqm_kernels.hamiltonian_lowering_preregistration import load_preregistration
from tt_rqm_kernels.su2_benchmark_release import validate_release as validate_su2_release

QMUL_RELEASE = Path("benchmarks/manifests/wormhole-qmul-level2.json")
SU2_RELEASE = Path("benchmarks/manifests/wormhole-su2-compose-level2.json")
V3_QUALIFICATION = Path("benchmarks/processed/wormhole-su2-compose-v3-stability-qualification.json")
STATUS_FILES = (
    "README.md",
    "plan.md",
    "docs/index.md",
    "docs/benchmarks/index.md",
    "docs/benchmarks/su2-compose-bench.md",
    "docs/benchmarks/su2-compose-claim-policy.md",
    "docs/hamiltonian-evolution-roadmap.md",
    "docs/tenstorrent-landing.md",
    "docs/collaboration-map.md",
)


class RepositoryClaimsError(ValueError):
    """Raised when repository claim surfaces contradict protected releases."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RepositoryClaimsError(message)


def validate_repository_claims(
    repo_root: Path,
    *,
    release_root: Path | None = None,
) -> dict[str, Any]:
    """Validate current release facts and the designated status documents."""

    repo_root = repo_root.resolve()
    evidence_root = (release_root or repo_root).resolve()
    qmul = validate_qmul_release(evidence_root / QMUL_RELEASE, repo_root=evidence_root)
    su2 = validate_su2_release(evidence_root / SU2_RELEASE, repo_root=evidence_root)
    qualification = json.loads((evidence_root / V3_QUALIFICATION).read_text(encoding="utf-8"))
    h2a = load_preregistration(
        evidence_root / "benchmarks" / "manifests" / "hamiltonian-lowering-h2a-preregistration.json"
    )
    expected_claim = {
        "level": 2,
        "name": "stable_one_device_performance",
        "public_session_count": 3,
        "stable_benchmark": True,
    }
    _require(qmul.get("claim") == expected_claim, "qmul current release is not Level 2 stable")
    _require(su2.get("claim") == expected_claim, "SU2 current release is not Level 2 stable")
    _require(len(qmul.get("sessions", ())) == 3, "qmul release must contain three sessions")
    _require(len(su2.get("sessions", ())) == 3, "SU2 release must contain three sessions")
    _require(
        "no_stable_fused_unfused_comparison_claim" in su2.get("nonclaims", ()),
        "SU2 release must remain fused-only",
    )
    _require(
        qualification.get("qualification_passed") is True
        and qualification.get("stable_benchmark") is True,
        "v3 qualification is not the passing stable aggregate",
    )
    for session in (*qmul["sessions"], *su2["sessions"]):
        report_path = evidence_root / session["performance_report"]
        report = json.loads(report_path.read_text(encoding="utf-8"))
        _require(
            report.get("stable_benchmark") is False,
            f"individual source session was promoted: {session['id']}",
        )
    _require(h2a["status"] == "pre_hardware", "H2A status must remain pre_hardware")

    from scripts.repo_status import _h2a_foundation_status, _su2_stability_status

    _require(
        _su2_stability_status(evidence_root)[0] == "established",
        "repo_status.py does not report SU2 stability as established",
    )
    _require(
        _h2a_foundation_status(evidence_root)[0] == "implementation-ready reference foundation",
        "repo_status.py does not report the H2A reference foundation",
    )
    documents = _load_status_documents(repo_root)
    _validate_status_surfaces(documents)
    return {
        "schema": "tt-rqm-repository-claims-validation.v1",
        "qmul": expected_claim,
        "su2": {**expected_claim, "scope": "fused_only"},
        "h2a": {"status": "pre_hardware", "target_claim_level": 0},
        "status_files": list(STATUS_FILES),
    }


def _load_status_documents(repo_root: Path) -> dict[str, str]:
    documents: dict[str, str] = {}
    for relative in STATUS_FILES:
        path = repo_root / relative
        _require(path.is_file(), f"missing status surface: {relative}")
        documents[relative] = path.read_text(encoding="utf-8")
    return documents


def _validate_status_surfaces(documents: dict[str, str]) -> None:
    plan = _status_block(documents["plan.md"])
    required_plan = (
        "qmul: Claim Level 2 stable one-device performance",
        "SU2ComposeBench H1: Claim Level 2 stable one-device **fused-only** performance",
        "Every individual qmul and H1 source-session report remains `stable_benchmark=false`",
        "historical H1 v2 fused/unfused campaign is retained and non-qualifying",
        "Active implementation milestone: H2A device-side two-level Hamiltonian coefficient lowering foundation. Hardware execution is not yet implemented.",
        "Future integration: H2B device-resident H2A lowering directly feeding the protected fused H1 composition path.",
    )
    for marker in required_plan:
        _require(marker in " ".join(plan.split()), f"plan status marker missing: {marker}")

    required_markers = {
        "README.md": (
            "SU2ComposeBench` is fused-only",
            "| SU2ComposeBench | fused time-ordered SU(2) composition on one Wormhole device | Level 2 | `true` |",
            "no H2 hardware execution is claimed",
        ),
        "docs/index.md": ("fused-only v3 campaign established the public Claim Level 2 release",),
        "docs/benchmarks/index.md": (
            "Claim Level 2 — stable one-device fused performance",
            "H2A is the active technical milestone",
        ),
        "docs/benchmarks/su2-compose-bench.md": (
            "Level 2: stable one-device fused performance",
            "retained v2 campaign remains historical and non-qualifying",
            "every source session remains",
        ),
        "docs/benchmarks/su2-compose-claim-policy.md": (
            "Level 3 is the first level that may qualify a matched-scope fused/unfused comparison",
            "separate v3 fused-only campaign is the current Claim Level 2 release",
        ),
        "docs/hamiltonian-evolution-roadmap.md": (
            "H1: completed fused-composition baseline",
            "H2A: device-side coefficient lowering",
            "no H2 hardware result exists yet",
            "H2B: future resident lowering plus H1 composition",
        ),
        "docs/tenstorrent-landing.md": (
            "SU2ComposeBench fused H1 path: Claim Level 2 stable one-device release",
            "there is no H2 hardware result",
        ),
        "docs/collaboration-map.md": (
            "SU2ComposeBench` has a fused-only Claim Level 2 release",
            "H2A coefficient lowering is now the active implementation foundation",
        ),
    }
    for relative, markers in required_markers.items():
        text = documents[relative]
        normalized = " ".join(text.split())
        for marker in markers:
            _require(
                marker in normalized,
                f"status marker missing from {relative}: {marker}",
            )

    combined = "\n".join(documents.values()).lower()
    forbidden = (
        "stable fused/unfused comparison: established",
        "v3 fused/unfused stable release",
        "h2a hardware implementation: complete",
        "h2a hardware execution is established",
        "cpu acceleration: established",
        "application acceleration: established",
    )
    for phrase in forbidden:
        _require(phrase not in combined, f"forbidden claim escalation: {phrase}")


def _status_block(plan: str) -> str:
    start_marker = "<!-- repository-claims:start -->"
    end_marker = "<!-- repository-claims:end -->"
    _require(start_marker in plan and end_marker in plan, "plan claim-status block is missing")
    return plan.split(start_marker, 1)[1].split(end_marker, 1)[0]
