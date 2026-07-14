from __future__ import annotations

import copy
import json
from pathlib import Path
import shlex
import sys

import pytest

from tt_rqm_kernels.benchmark_integrity import IntegrityError, validate_report
from tt_rqm_kernels.backends.tenstorrent.qmul_persistent import (
    DEVICE,
    IMPLEMENTATION_CLASS,
    PERSISTENT_METRICS_SCHEMA,
    PERSISTENT_PROTOCOL,
    run_persistent_qmul,
    validate_persistent_metrics,
    validate_persistent_report,
)

FIXTURE = Path("tests/fixtures/qmul_persistent_candidate.py").resolve()


def test_persistent_fixture_conformance_validates_whole_output() -> None:
    report = run_persistent_qmul(
        command=shlex.join((sys.executable, str(FIXTURE))),
        benchmark_stage="conformance",
        methodology_note="Hardware-independent protocol fixture.",
    )

    validate_persistent_report(report)
    result = report["results"][0]
    assert result["items"] == 128
    assert result["correctness"]["validated_values"] == 512
    assert result["correctness"]["failing_values"] == 0
    assert result["correctness"]["nonfinite_values"] == 0
    assert result["implementation_class"] == IMPLEMENTATION_CLASS
    assert report["lifecycle"]["create_count"] == 1
    assert report["lifecycle"]["close_count"] == 1


def test_persistent_fixture_performance_is_one_session_and_strict_sweep() -> None:
    report = run_persistent_qmul(
        command=shlex.join((sys.executable, str(FIXTURE))),
        benchmark_stage="performance",
        methodology_note="Hardware-independent protocol fixture.",
    )

    assert report["case_items"] == [4096, 65536, 262144]
    assert [result["timing"]["repetitions"] for result in report["results"]] == [
        10,
        10,
        10,
    ]
    assert report["lifecycle"] == {
        "device_count": 1,
        "device_id": 0,
        "create_count": 1,
        "close_count": 1,
    }


def test_strict_metrics_reject_stale_label_device_samples_hash_and_timing() -> None:
    manifest, metrics = _minimal_contract()
    valid = lambda payload: validate_persistent_metrics(
        payload,
        manifest,
        candidate_sha256="a" * 64,
        host_process_s=1.0,
    )
    valid(metrics)
    mutations = [
        ("label", lambda value: value.__setitem__("execution_kind", "emulation")),
        ("device", lambda value: value["lifecycle"].__setitem__("device_id", 1)),
        ("samples", lambda value: value["cases"][0]["timings_s"].__setitem__("samples", [])),
        ("hash", lambda value: value["provenance"].__setitem__("candidate_sha256", "b" * 64)),
        ("stable", lambda value: value.__setitem__("stable_benchmark", True)),
        ("timing", lambda value: value["session_timings_s"].__setitem__("candidate_session", 2.0)),
    ]
    for _, mutate in mutations:
        changed = copy.deepcopy(metrics)
        mutate(changed)
        with pytest.raises(IntegrityError):
            valid(changed)


def test_persistent_source_owns_one_device_lifecycle_and_reuses_audited_kernels() -> None:
    source = Path(
        "experimental/tt_metalium_qmul/src/qmul_multicore_persistent_candidate.cpp"
    ).read_text()
    cmake = Path("experimental/tt_metalium_qmul/CMakeLists.txt").read_text()

    assert source.count("MeshDevice::create_unit_mesh") == 1
    assert '#include "qmul_multicore_candidate.cpp"' in source
    assert "class PersistentDeviceSession" in source
    assert "~PersistentDeviceSession()" in source
    assert "if (device_ && !closed_)" in source
    assert "for (const auto& spec : manifest.at(\"cases\"))" in source
    assert "session.close()" in source
    assert "device_id != 0" in source
    assert "multicore_tensix_sfpu_qmul_persistent" in source
    assert "tt_rqm_metalium_qmul_multicore_persistent_candidate" in cmake


def test_persistent_padding_boundaries_cover_partial_and_multicore_tiles() -> None:
    for items in (1, 128, 1023, 1024, 1025, 4096, 65537, 262144):
        tiles = (items + 1023) // 1024
        padded = tiles * 1024
        assert padded >= items
        assert padded - items < 1024
        assert min(tiles, 56) >= 1


def test_existing_stage_a_and_first_stage_b_reports_remain_valid() -> None:
    paths = (
        "reports/tt_hardware_qmul_quickstart.json",
        "reports/tt_hardware_qmul_stage_b_candidate_conformance.json",
        "reports/tt_hardware_qmul_stage_b_performance.json",
    )
    for path in paths:
        validate_report(json.loads(Path(path).read_text()))


def test_stability_methodology_is_preregistered_and_preserves_nonclaims() -> None:
    methodology = Path("docs/stage-b-stability-methodology.md").read_text()
    readme = Path("experimental/tt_metalium_qmul/README.md").read_text()

    assert "10.4825%" in methodology
    assert "5.0000%" in methodology
    assert "at least three independent" in methodology
    assert "initial persistent" in methodology
    assert "must remain `false`" in methodology
    assert "future CPU" in readme
    assert "Device 1 is out of scope" in readme


def _minimal_contract() -> tuple[dict, dict]:
    work = {
        "device_count": 1,
        "device_id": 0,
        "core_count": 1,
        "component_tiles": 1,
        "grid_x": 8,
        "grid_y": 7,
        "available_core_count": 56,
        "layout": "planar_float32_tiles_32x32",
        "work_split": "row_major",
        "arithmetic_path": "tensix_compute_sfpu",
    }
    case = {
        "case_id": "case-128",
        "items": 128,
        "iterations": 1,
        "warmup": 0,
        "samples": 1,
        "inputs": {"a_sha256": "1" * 64, "b_sha256": "2" * 64},
    }
    manifest = {"cases": [case]}
    metrics = {
        "schema": PERSISTENT_METRICS_SCHEMA,
        "protocol": PERSISTENT_PROTOCOL,
        "device": DEVICE,
        "dtype": "float32",
        "execution_kind": "hardware",
        "implementation_class": IMPLEMENTATION_CLASS,
        "performance_eligible": True,
        "stable_benchmark": False,
        "lifecycle": {
            "device_count": 1,
            "device_id": 0,
            "create_count": 1,
            "close_count": 1,
        },
        "session_timings_s": {
            "device_create": 0.01,
            "device_close": 0.01,
            "candidate_session": 0.2,
        },
        "provenance": {
            "chip_type": "n300",
            "tt_metal_commit": "d" * 40,
            "compiler_version": "c++",
            "runtime_version": "metalium",
            "build_id": "a" * 64,
            "timer_scope": "persistent test",
            "candidate_sha256": "a" * 64,
            "repository_commit": "c" * 40,
        },
        "cases": [
            {
                **{key: case[key] for key in ("case_id", "items", "iterations", "warmup", "samples")},
                "input_identity": {
                    "a_sha256": "1" * 64,
                    "b_sha256": "2" * 64,
                },
                "output_identity": {"fnv1a64": "0" * 16, "value_count": 512},
                "timings_s": {
                    "buffer_allocation": 0.001,
                    "program_build": 0.001,
                    "h2d": 0.001,
                    "prewarm_sync": 0.001,
                    "warmup": 0.001,
                    "samples": [0.001],
                    "d2h": 0.001,
                    "cleanup": 0.001,
                },
                "work": work,
            }
        ],
    }
    return manifest, metrics
