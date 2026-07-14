from __future__ import annotations

import copy
from pathlib import Path

import pytest

from tt_rqm_kernels.backends.tenstorrent.su2_compose_persistent import (
    DEVICE,
    IMPLEMENTATION,
    METRICS_SCHEMA,
    PROTOCOL,
    TT_METAL_COMMIT,
    _case_specs,
    validate_su2_metrics,
)
from tt_rqm_kernels.benchmark_integrity import IntegrityError


PACKAGE = Path("experimental/tt_metalium_su2_compose")


def _manifest_and_metrics() -> tuple[dict[str, object], dict[str, object]]:
    case = {
        "case_id": "case", "B": 2048, "K": 8, "repeat_count": 1,
        "warmup_pairs": 0, "samples": 1,
        "inputs": {"rotors_sha256": "a" * 64, "phases_sha256": "b" * 64},
    }
    manifest = {"cases": [case]}
    metrics = {
        "schema": METRICS_SCHEMA, "protocol": PROTOCOL, "device": DEVICE,
        "dtype": "float32", "execution_kind": "hardware",
        "implementation_class": IMPLEMENTATION, "performance_eligible": False,
        "stable_benchmark": False,
        "lifecycle": {"device_count": 1, "device_id": 0, "create_count": 1, "close_count": 1},
        "session_timings_s": {"device_create": 1.0, "device_close": 0.1, "candidate_session": 2.0},
        "provenance": {
            "chip_type": "wormhole_b0", "tt_metal_commit": TT_METAL_COMMIT,
            "compiler_version": "gcc", "runtime_version": "runtime", "build_id": "build",
            "candidate_sha256": "c" * 64, "repository_commit": "d" * 40,
        },
        "cases": [{
            **case,
            "timings_s": {"fused_samples": [0.1], "unfused_samples": [0.2], "paired_order": ["fused_first"]},
            "work": {"device_count": 1, "device_id": 0, "core_count": 2,
                     "available_core_count": 56, "fused_dispatches_per_chain": 1,
                     "unfused_dispatches_per_chain": 7},
        }],
    }
    return manifest, metrics


def test_hardware_package_contains_separate_fused_and_unfused_paths() -> None:
    required = (
        "src/su2_compose_candidate.cpp", "kernels/su2_fused_reader.cpp",
        "kernels/su2_fused_compute.cpp", "kernels/su2_fused_writer.cpp",
        "kernels/su2_unfused_reader.cpp", "kernels/su2_unfused_compute.cpp",
        "kernels/su2_unfused_writer.cpp", "kernels/su2_sfpu.h",
    )
    for relative in required:
        assert (PACKAGE / relative).is_file()


def test_arithmetic_is_confined_to_compute_sfpu_sources() -> None:
    readers_and_writers = [
        *(PACKAGE / "kernels").glob("*reader.cpp"),
        *(PACKAGE / "kernels").glob("*writer.cpp"),
    ]
    for path in readers_and_writers:
        source = path.read_text()
        assert "su2_product_sfpu" not in source
        assert "su2_quaternion_component" not in source
        assert "su2_phase_component" not in source
    assert "su2_quaternion_component" in (PACKAGE / "kernels/su2_compute_common.h").read_text()
    assert "fused_accumulator_storage" in (PACKAGE / "src/su2_compose_candidate.cpp").read_text()


def test_preregistered_case_specs_are_exact() -> None:
    assert _case_specs("conformance") == ((32, 8, 1, 0, 1), (2048, 8, 1, 0, 1))
    performance = _case_specs("performance")
    assert len(performance) == 8
    assert performance[0] == (32768, 8, 10, 2, 10)
    assert performance[-1] == (65536, 128, 1, 2, 10)


def test_strict_metrics_accept_valid_pre_eligibility_conformance() -> None:
    manifest, metrics = _manifest_and_metrics()
    validated = validate_su2_metrics(metrics, manifest, "c" * 64, 3.0)
    assert validated["performance_eligible"] is False


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("lifecycle", "device_count"), 2),
        (("provenance", "tt_metal_commit"), "wrong"),
        (("provenance", "candidate_sha256"), "0" * 64),
        (("cases", 0, "work", "core_count"), 1),
        (("cases", 0, "work", "unfused_dispatches_per_chain"), 8),
        (("cases", 0, "timings_s", "paired_order"), []),
    ],
)
def test_strict_metrics_reject_contract_drift(path: tuple[object, ...], value: object) -> None:
    manifest, metrics = _manifest_and_metrics()
    changed = copy.deepcopy(metrics)
    target: object = changed
    for part in path[:-1]:
        target = target[part]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]
    with pytest.raises(IntegrityError):
        validate_su2_metrics(changed, manifest, "c" * 64, 3.0)
