#!/usr/bin/env python3
"""H2B external-protocol fixture with deliberate fault injection."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.hamiltonian_evolution_external_reference import main as run_reference


def main() -> int:
    status = run_reference()
    if status:
        return status
    work = Path(os.environ["TT_RQM_H2B_DIR"])
    manifest = json.loads(Path(os.environ["TT_RQM_H2B_MANIFEST"]).read_text())
    metrics_path = work / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    if os.environ.get("TT_RQM_H2B_TEST_HARDWARE") == "1":
        metrics["execution_label"] = "hardware"
        metrics["candidate_metadata"] = _hardware_metadata()
    rotor_path = work / manifest["outputs"]["final_rotors"]
    phase_path = work / manifest["outputs"]["final_phases"]
    fault = os.environ.get("TT_RQM_H2B_TEST_FAULT")
    if fault == "malformed_metrics":
        metrics_path.write_text("{")
        return 0
    if fault == "wrong_lane_order":
        metrics["final_rotor_lane_order"] = ["x", "w", "y", "z"]
    elif fault == "truncate_rotor":
        rotor_path.write_bytes(rotor_path.read_bytes()[:-4])
    elif fault == "truncate_phase":
        phase_path.write_bytes(phase_path.read_bytes()[:-4])
    elif fault == "reorder":
        payload = rotor_path.read_bytes()
        rotor_path.write_bytes(payload[16:32] + payload[0:16] + payload[32:])
    elif fault in {"nan", "inf"}:
        payload = bytearray(rotor_path.read_bytes())
        payload[0:4] = struct.pack("<f", math.nan if fault == "nan" else math.inf)
        rotor_path.write_bytes(payload)
    elif fault == "device_metadata":
        metrics["candidate_metadata"]["device_id"] = 1
    elif fault == "host_round_trip":
        metrics["candidate_metadata"]["host_round_trip_count"] = 1
    elif fault == "intermediate_d2h":
        metrics["candidate_metadata"]["intermediate_d2h_count"] = 1
    elif fault == "intermediate_h2d":
        metrics["candidate_metadata"]["intermediate_h2d_count"] = 1
    elif fault == "tt_metal_commit":
        metrics["candidate_metadata"]["tt_metal_commit"] = "f" * 40
    elif fault == "stable":
        metrics["stable_benchmark"] = True
    elif fault == "performance":
        metrics["performance_eligible"] = True
    elif fault == "claim":
        metrics["claim_level"] = 0
    metrics_path.write_text(json.dumps(metrics))
    return 0


def _hardware_metadata() -> dict[str, object]:
    return {
        "implementation_class": "two_program_device_resident_h2b",
        "candidate_sha256": "a" * 64,
        "source_commit": "b" * 40,
        "source_tree_clean": False,
        "source_bundle_sha256": "c" * 64,
        "tt_metal_commit": "dd2849b5bc6b7a5d38a9eafbeba31ef8d530f8d4",
        "compiler_version": "c++",
        "runtime_version": "tt-metal-pinned",
        "device_arch": "wormhole_b0",
        "device_count": 1,
        "device_id": 0,
        "device_create_count": 1,
        "device_close_count": 1,
        "program_count": 2,
        "h2a_core_count": 1,
        "h1_core_count": 1,
        "input_layout": "step-major six-plane tiles",
        "intermediate_layout": "step-major six-plane tiles",
        "output_layout": "six-plane final tiles",
        "intermediate_storage": "device_dram",
        "device_resident_intermediate": True,
        "intermediate_d2h_count": 0,
        "intermediate_h2d_count": 0,
        "host_round_trip_count": 0,
        "h2a_arithmetic_path": "compensated Tensix SFPU",
        "h1_arithmetic_path": "fused Tensix SFPU",
        "composition_order": "K-1 ... 0",
        "automatic_normalization": False,
    }


if __name__ == "__main__":
    raise SystemExit(main())
