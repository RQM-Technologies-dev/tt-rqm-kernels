from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_physical_ai_pose_stream_demo_reports_reference_metrics() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "examples/physical_ai_pose_stream.py",
            "--items",
            "32",
            "--iters",
            "1",
            "--warmup",
            "0",
            "--seed",
            "0",
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    report = json.loads(completed.stdout)
    assert report["schema"] == "tt-rqm-pose-stream-demo.v1"
    assert report["backend"] == "torch"
    assert report["device"] == "cpu"
    assert report["items"] == 32
    assert report["structured_shape"] == (
        "orientation=[32, 4], body_vector=[32, 3], world_vector=[32, 3]"
    )
    assert report["unit_rotor_max_abs_error"] < 1e-5
    assert report["norm_preservation_max_abs"] < 1e-5
    assert "not Tenstorrent hardware performance" in " ".join(report["notes"])
