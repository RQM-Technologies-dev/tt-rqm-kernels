from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tt_rqm_kernels.qmul_hardware_evidence import generate_all


ROOT = Path(__file__).resolve().parents[1]


def _hashes(paths: list[Path]) -> dict[str, str]:
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def test_hardware_evidence_generation_is_deterministic() -> None:
    first = generate_all(ROOT)
    before = _hashes(first)
    second = generate_all(ROOT)
    assert before == _hashes(second)


def test_processed_evidence_preserves_claim_boundaries() -> None:
    processed = ROOT / "benchmarks/processed"
    index = json.loads((processed / "wormhole-qmul-hardware-evidence-index.json").read_text())
    assert index["current_public_claim_level"] == 2
    assert index["stability_qualification_passed"] is True
    assert index["public_claim_updated"] is True

    profiler = json.loads((processed / "wormhole-qmul-profiler-and-ceilings.json").read_text())
    fp32 = next(value for value in profiler["ceilings"] if value["kind"] == "compute_fp32_sfpu")
    assert fp32["value"] is None
    assert "not available" in fp32["status"]

    saturation = json.loads((processed / "wormhole-qmul-saturation.json").read_text())
    assert saturation["memory_preflight"]["safe"] is True
    assert saturation["occupancy_knee"] == {"items": 57344, "component_tiles": 56, "actual_cores": 56}
    assert all(row["correctness_passed"] for row in saturation["rows"])


def test_core_scaling_never_allocates_idle_cores() -> None:
    path = ROOT / "benchmarks/processed/wormhole-qmul-core-scaling.json"
    payload = json.loads(path.read_text())
    assert all(row["requested_cores"] == row["actual_cores"] for row in payload["rows"])
    assert all(row["actual_cores"] <= row["component_tiles"] for row in payload["rows"])
