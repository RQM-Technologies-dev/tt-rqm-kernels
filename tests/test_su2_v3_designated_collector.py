from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "collect_su2_compose_v3_designated",
    ROOT / "scripts/collect_su2_compose_v3_designated.py",
)
assert SPEC is not None and SPEC.loader is not None
collector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collector)


def _preregistration() -> dict[str, object]:
    return {
        "status": "frozen_before_designated_session_1",
        "candidate": {
            "sha256": "a" * 64,
            "source_commit": "source",
            "source_tree_sha256": "b" * 64,
            "tt_metal_commit": "metal",
            "compiler_version": "compiler",
            "runtime_version": "runtime",
        },
        "host_contract": {
            "cpu_model": "cpu",
            "inherited_cpu_affinity": [0],
            "requested_candidate_cpu_affinity": [24, 25, 26, 27],
            "process_nice": 0,
            "cpu_governors": ["schedutil"],
            "numa_nodes": ["node0"],
            "profiler_watcher_debug_disabled": True,
            "timing_environment": {},
        },
        "pilot_repeat_plan": "repeat-plan.json",
    }


def test_preflight_rejects_host_contract_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = tmp_path / "candidate"
    candidate.write_text("candidate")
    candidate.chmod(0o755)
    preregistration = _preregistration()
    monkeypatch.setattr(collector, "load_stability_preregistration", lambda *args, **kwargs: preregistration)
    monkeypatch.setattr(collector, "sha256_file", lambda path: "a" * 64)
    def fake_git(root: Path, *args: str) -> str:
        if args[-1] == "--porcelain":
            return ""
        return "metal" if root.name == "metal" else "source"

    monkeypatch.setattr(collector, "_git_value", fake_git)
    monkeypatch.setattr(collector, "validate_source_tree", lambda **kwargs: "b" * 64)
    monkeypatch.setattr(collector, "_compiler_version", lambda: "compiler")
    monkeypatch.setattr(collector, "_timing_environment", lambda: {})
    monkeypatch.setattr(
        collector,
        "capture_host_state",
        lambda **kwargs: {
            "cpu_model": "wrong-cpu",
            "inherited_cpu_affinity": [0],
            "requested_candidate_cpu_affinity": [24, 25, 26, 27],
            "process_nice": 0,
            "cpu_governors": ["schedutil"],
            "numa_nodes": ["node0"],
        },
    )
    monkeypatch.setattr(collector, "validate_device_health", lambda *args, **kwargs: {})

    with pytest.raises(Exception, match="host contract differs for cpu_model"):
        collector.preflight(
            command=str(candidate),
            repository_root=tmp_path / "repository",
            execution_source_root=tmp_path,
            tt_metal_root=tmp_path / "metal",
            tt_smi_command="true",
            preregistration_path=Path("manifest.json"),
        )


def test_designated_collector_uses_only_frozen_ids_and_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preregistration = _preregistration()
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        collector,
        "preflight",
        lambda **kwargs: {
            "candidate_sha256": "a" * 64,
            "source_commit": "source",
            "source_tree_sha256": "b" * 64,
            "tt_metal_commit": "metal",
            "compiler_version": "compiler",
            "runtime_version": "runtime",
        },
    )
    monkeypatch.setattr(collector, "load_stability_preregistration", lambda *args, **kwargs: preregistration)
    monkeypatch.setattr(
        collector,
        "load_v3_pilot_repeat_counts",
        lambda *args, **kwargs: {
            (32768, 8): 267, (8192, 32): 90, (2048, 128): 24, (512, 512): 7,
            (1024, 128): 24, (4096, 128): 24, (16384, 128): 24, (65536, 128): 12,
        },
    )

    def fake_collect(**kwargs):
        captured.update(kwargs)
        return kwargs["session_dir"]

    monkeypatch.setattr(collector, "collect_su2_session", fake_collect)
    monkeypatch.setattr(
        sys,
        "argv",
        ["collector", "--command", str(tmp_path / "candidate"), "--session-id", "su2-v3-level2-session-1", "--repository-root", str(tmp_path), "--tt-metal-root", str(tmp_path), "--output-root", str(tmp_path / "output")],
    )

    assert collector.main() == 0
    assert captured["designated_stability_session"] is True
    assert captured["benchmark_mode"] == collector.FUSED_STABILITY
    assert captured["benchmark_stage"] == "performance"
    assert captured["cpu_affinity"] == collector.FROZEN_CPU_AFFINITY
    assert captured["isolate_runtime_cache"] is True
    assert captured["case_specs"] == (
        (32768, 8, 267, 5, 10), (8192, 32, 90, 5, 10), (2048, 128, 24, 5, 10), (512, 512, 7, 5, 10),
        (1024, 128, 24, 5, 10), (4096, 128, 24, 5, 10), (16384, 128, 24, 5, 10), (65536, 128, 12, 5, 10),
    )


def test_designated_collector_rejects_nonfrozen_session_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["collector", "--command", "/candidate", "--session-id", "retry"])
    with pytest.raises(SystemExit, match="2"):
        collector.main()
