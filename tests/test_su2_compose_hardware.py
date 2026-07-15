from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from tt_rqm_kernels.backends.tenstorrent.su2_compose_persistent import (
    DEVICE,
    FUSED_STABILITY,
    IMPLEMENTATION,
    METRICS_SCHEMA,
    PROTOCOL,
    TT_METAL_COMMIT,
    _case_specs,
    fused_stability_case_specs,
    run_su2_compose,
    validate_su2_metrics,
)
from tt_rqm_kernels.benchmark_integrity import IntegrityError
from tt_rqm_kernels.su2_hardware_session import cache_inventory, parse_cpu_affinity


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


def test_audited_candidate_is_performance_eligible_but_not_stable() -> None:
    source = (PACKAGE / "src/su2_compose_candidate.cpp").read_text()
    assert "constexpr bool kPerformanceEligible = true;" in source
    assert '{"stable_benchmark", false}' in source


def test_fused_stability_candidate_does_not_construct_unfused_workload() -> None:
    source = (PACKAGE / "src/su2_compose_candidate.cpp").read_text()
    assert 'benchmark_mode == "fused_stability"' in source
    assert "if (!fused_only) unfused.emplace" in source
    assert 'work["unfused_dispatches_per_chain"]' in source
    assert '"synchronization_boundaries_per_sample", fused_only ? 1' in source


def test_candidate_command_cannot_hide_binary_behind_env_wrapper() -> None:
    with pytest.raises(IntegrityError, match="must name the candidate directly"):
        run_su2_compose(
            command="env TT_METAL_RUNTIME_ROOT=/tmp candidate",
            stage="conformance",
            methodology_note="wrapper identity rejection",
        )


def test_preregistered_case_specs_are_exact() -> None:
    assert _case_specs("conformance") == ((32, 8, 1, 0, 1), (2048, 8, 1, 0, 1))
    performance = _case_specs("performance")
    assert len(performance) == 8
    assert performance[0] == (32768, 8, 10, 2, 10)
    assert performance[-1] == (65536, 128, 1, 2, 10)
    repeat_counts = {(batch, steps): index + 1 for index, (batch, steps) in enumerate(
        ((32768, 8), (8192, 32), (2048, 128), (512, 512),
         (1024, 128), (4096, 128), (16384, 128), (65536, 128))
    )}
    fused = fused_stability_case_specs(repeat_counts)
    assert fused[0] == (32768, 8, 1, 5, 10)
    assert fused[-1] == (65536, 128, 8, 5, 10)


def test_strict_metrics_accept_valid_pre_eligibility_conformance() -> None:
    manifest, metrics = _manifest_and_metrics()
    validated = validate_su2_metrics(metrics, manifest, "c" * 64, 3.0)
    assert validated["performance_eligible"] is False


def test_strict_metrics_accept_fused_only_stability_surface() -> None:
    case = {
        "case_id": "case", "B": 2048, "K": 8, "repeat_count": 20,
        "warmup_count": 5, "samples": 10,
        "inputs": {"rotors_sha256": "a" * 64, "phases_sha256": "b" * 64},
    }
    manifest = {"benchmark_mode": FUSED_STABILITY, "cases": [case]}
    metrics = {
        "schema": METRICS_SCHEMA, "protocol": PROTOCOL, "device": DEVICE,
        "dtype": "float32", "execution_kind": "hardware", "benchmark_mode": FUSED_STABILITY,
        "implementation_class": IMPLEMENTATION, "performance_eligible": True,
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
            "timings_s": {"fused_samples": [0.0375] * 10},
            "work": {"device_count": 1, "device_id": 0, "core_count": 2,
                     "available_core_count": 56, "fused_dispatches_per_chain": 1,
                     "synchronization_boundaries_per_sample": 1},
        }],
    }
    assert validate_su2_metrics(metrics, manifest, "c" * 64, 3.0)["benchmark_mode"] == FUSED_STABILITY
    changed = copy.deepcopy(metrics)
    changed["cases"][0]["timings_s"]["unfused_samples"] = [0.1] * 10
    with pytest.raises(IntegrityError):
        # Candidate metrics may not smuggle comparison data into Level 2.
        validate_su2_metrics(changed, manifest, "c" * 64, 3.0)


def test_cpu_affinity_and_cache_inventory_are_deterministic(tmp_path: Path) -> None:
    assert parse_cpu_affinity("2-4,7") == frozenset({2, 3, 4, 7})
    with pytest.raises(IntegrityError):
        parse_cpu_affinity("4-2")
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "kernel.bin").write_bytes(b"kernel")
    first = cache_inventory(cache)
    second = cache_inventory(cache)
    first.pop("cache_root")
    second.pop("cache_root")
    assert first == second


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


def test_committed_conformance_release_hashes_and_claim_are_valid() -> None:
    release = json.loads(Path("benchmarks/manifests/su2-compose-conformance.json").read_text())
    assert release["claim"] == {"level": 0, "name": "silicon_conformance", "stable_benchmark": False}
    for artifact in release["artifacts"]:
        assert hashlib.sha256(Path(artifact["path"]).read_bytes()).hexdigest() == artifact["sha256"]


def test_public_hardware_claim_is_immediately_qualified() -> None:
    for path in (Path("README.md"), Path("docs/tenstorrent-landing.md"), Path("docs/benchmarks/su2-compose-bench.md")):
        text = path.read_text()
        claim = text.index("RQM runs fused time-ordered SU(2) evolution")
        qualification = text.index("on the CPU", claim)
        assert qualification - claim < 500


def test_readme_summarizes_two_aggregate_benchmark_releases() -> None:
    text = Path("README.md").read_text()
    table = text.split("| Evidence | Implementation | Claim | Stable benchmark |", 1)[1]
    table = table.split("\n\n", 1)[0].strip()
    assert table.splitlines() == [
        "|---|---|---|---|",
        "| qmul | multicore Tensix compute/SFPU on one Wormhole device; Stage A baseline retained | Level 2 | `true` |",
        "| SU2ComposeBench H1 | fused time-ordered SU(2) composition on one Wormhole device | Level 2 | `true` |",
    ]
    assert "Stage A qmul conformance" not in table
    assert "Stage B qmul" not in table
    assert "Three designated v3 cold-start N300 sessions passed" in text
    assert "stable_benchmark=true" in text
    assert "no fused/unfused comparison or" in text
