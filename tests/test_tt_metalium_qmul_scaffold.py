from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


def test_tt_metalium_environment_check_missing_sdk_is_clear() -> None:
    env = _without_tt_metal_env()
    completed = subprocess.run(
        [
            sys.executable,
            "experimental/tt_metalium_qmul/check_environment.py",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 2
    assert "TT-Metalium SDK unavailable" in completed.stderr
    assert "TT_METAL_HOME" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_tt_metalium_placeholder_without_external_env_fails_cleanly() -> None:
    env = _without_external_qmul_env()
    completed = subprocess.run(
        [
            sys.executable,
            "experimental/tt_metalium_qmul/run_candidate.py",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 2
    assert "external-qmul environment missing" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_tt_metalium_build_placeholder_missing_sdk_is_clear() -> None:
    env = _without_tt_metal_env()
    completed = subprocess.run(
        [
            sys.executable,
            "experimental/tt_metalium_qmul/build_candidate.py",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 2
    assert "TT-Metalium SDK unavailable" in completed.stderr
    assert "Build stopped before configuring the candidate" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_tt_metalium_validation_wrapper_accepts_reference_command() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "experimental/tt_metalium_qmul/validate_candidate.py",
            "--candidate-command",
            "python scripts/qmul_external_reference.py",
            "--items",
            "32",
            "--iters",
            "1",
            "--warmup",
            "0",
            "--format",
            "json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(completed.stdout)
    assert report["schema"] == "structuredbench.v1"
    assert report["backend"] == "external-qmul"
    assert report["protocol"] == "tt-rqm-external-qmul.v1"
    assert report["results"][0]["structured_shape"] == "[32, 4]"
    assert report["results"][0]["scalar_reference_max_abs_error"] < 1e-4


def test_tt_metalium_validation_wrapper_placeholder_fails_without_sdk() -> None:
    env = _without_tt_metal_env()
    completed = subprocess.run(
        [
            sys.executable,
            "experimental/tt_metalium_qmul/validate_candidate.py",
            "--items",
            "32",
            "--iters",
            "1",
            "--warmup",
            "0",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode != 0
    assert "TT-Metalium SDK unavailable" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert "No out.bin or metrics.json was written" not in completed.stdout
    assert "hardware performance" not in completed.stdout.lower()


def test_tt_metalium_source_candidate_files_exist() -> None:
    assert os.path.exists("experimental/tt_metalium_qmul/CMakeLists.txt")
    assert os.path.exists("experimental/tt_metalium_qmul/src/qmul_candidate.cpp")
    assert os.path.exists("experimental/tt_metalium_qmul/kernels/qmul_riscv.cpp")


def test_tt_metalium_multicore_candidate_files_exist() -> None:
    package = Path("experimental/tt_metalium_qmul")

    assert (package / "src/qmul_multicore_candidate.cpp").exists()
    assert (package / "kernels/qmul_multicore_reader.cpp").exists()
    assert (package / "kernels/qmul_multicore_compute.cpp").exists()
    assert (package / "kernels/qmul_multicore_writer.cpp").exists()
    assert (package / "kernels/qmul_sfpu.h").exists()


def test_tt_metalium_multicore_architecture_is_stage_b_candidate() -> None:
    host = Path("experimental/tt_metalium_qmul/src/qmul_multicore_candidate.cpp").read_text()
    reader = Path("experimental/tt_metalium_qmul/kernels/qmul_multicore_reader.cpp").read_text()
    compute = Path("experimental/tt_metalium_qmul/kernels/qmul_multicore_compute.cpp").read_text()
    writer = Path("experimental/tt_metalium_qmul/kernels/qmul_multicore_writer.cpp").read_text()
    sfpu = Path("experimental/tt_metalium_qmul/kernels/qmul_sfpu.h").read_text()

    assert "split_work_to_cores(grid, component_tiles, true)" in host
    assert "MeshDevice::create_unit_mesh(device_id)" in host
    assert "Stage B candidate is restricted to Wormhole device 0" in host
    assert "DataFormat::Float32" in host
    assert ".fp32_dest_acc_en = true" in host
    assert "multicore_tensix_sfpu_qmul" in host
    assert "constexpr bool kPerformanceEligible = false" in host
    assert "noc_async_read_page" in reader
    assert "noc_async_write_page" in writer
    assert "qmul_tiles_sfpu" in compute
    assert "out_w = aw * bw - ax * bx - ay * by - az * bz" in sfpu
    assert "out_x = aw * bx + ax * bw + ay * bz - az * by" in sfpu
    assert "out_y = aw * by - ax * bz + ay * bw + az * bx" in sfpu
    assert "out_z = aw * bz + ax * by - ay * bx + az * bw" in sfpu
    for data_movement_source in (reader, writer):
        assert "sfpi::" not in data_movement_source
        assert "out_w" not in data_movement_source


def test_tt_metalium_build_candidate_selection_defaults_to_scalar() -> None:
    module = _load_script("experimental/tt_metalium_qmul/build_candidate.py")

    assert module.CANDIDATE_TARGETS == {
        "scalar": "tt_rqm_metalium_qmul_candidate",
        "multicore": "tt_rqm_metalium_qmul_multicore_candidate",
    }
    assert module.DEFAULT_BINARY_NAME == module.CANDIDATE_TARGETS["scalar"]


def test_tt_metalium_candidate_is_stage_a_baseline_with_split_timing() -> None:
    source = Path("experimental/tt_metalium_qmul/src/qmul_candidate.cpp").read_text()

    assert "tt-rqm-external-qmul-metrics.v2" in source
    assert "scalar_riscv_correctness_baseline" in source
    assert '"performance_eligible\\\": false' in source
    assert '"timings_s\\\"' in source
    assert "build_workload(" in source
    assert "auto workload = build_workload(" in source
    assert "run_program_once(" not in source


def test_tt_metalium_readme_preserves_stage_a_hardware_record() -> None:
    readme = Path("experimental/tt_metalium_qmul/README.md").read_text()

    assert "Stage A N300 silicon\nconformance gate" in readme
    assert "tt-rqm-external-qmul-metrics.v2" in readme
    assert "f73221f014c2ea0c1ad9b44fbfd44c5492859943" in readme
    assert "efa529a59bc709fccb58d6134dedff3297f8fdaa" in readme
    assert "permanently ineligible for Stage B" in readme


def test_tt_metalium_readme_rejects_obsolete_scalar_guidance() -> None:
    readme = Path("experimental/tt_metalium_qmul/README.md").read_text()

    assert "It has not been run on Tenstorrent hardware." not in readme
    assert '"elapsed_s": 0.001' not in readme
    assert '"device": "candidate-device-label"' not in readme
    assert "For a larger report once the candidate is real" not in readme
    assert "--items 4096" not in readme
    assert "--iters 10" not in readme
    assert "--warmup 2" not in readme


def test_tt_metalium_build_accepts_lowercase_metalium_config(tmp_path: Path) -> None:
    module = _load_script("experimental/tt_metalium_qmul/build_candidate.py")
    prefix = tmp_path / "build_emule"
    prefix.mkdir()
    (prefix / "tt-metalium-config.cmake").write_text("# generated by tt-metal\n")
    (prefix / "Metalium.cmake").write_text("# exported targets\n")

    assert module._has_metalium_package(prefix)


def test_tt_metalium_build_prefers_installed_package_dir(tmp_path: Path) -> None:
    module = _load_script("experimental/tt_metalium_qmul/build_candidate.py")
    prefix = tmp_path / "build_emule"
    installed = prefix / "lib" / "cmake" / "tt-metalium"
    installed.mkdir(parents=True)
    (prefix / "tt-metalium-config.cmake").write_text("# stale root config\n")
    (installed / "tt-metalium-config.cmake").write_text("# installed config\n")
    (installed / "Metalium.cmake").write_text("# exported targets\n")

    assert module._has_metalium_package(prefix)
    assert module._metalium_package_dir(prefix) == installed.resolve()


def test_tt_metalium_build_rejects_incomplete_metalium_config(tmp_path: Path) -> None:
    module = _load_script("experimental/tt_metalium_qmul/build_candidate.py")
    prefix = tmp_path / "build_emule"
    prefix.mkdir()
    (prefix / "tt-metalium-config.cmake").write_text("# generated by tt-metal\n")

    assert not module._has_metalium_package(prefix)


def test_tt_metalium_build_prefix_expands_uninstalled_dependencies(tmp_path: Path) -> None:
    module = _load_script("experimental/tt_metalium_qmul/build_candidate.py")
    prefix = tmp_path / "build_emule"
    fmt_build = prefix / "_deps" / "fmt-build"
    cpm_modules = prefix / "CPM_modules"
    fmt_build.mkdir(parents=True)
    cpm_modules.mkdir()

    assert module._cmake_prefix_paths(prefix) == [prefix.resolve(), fmt_build.resolve()]
    assert module._cmake_module_paths(prefix) == [cpm_modules.resolve()]


def _without_tt_metal_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("TT_METAL_HOME", None)
    env.pop("TT_METALIUM_HOME", None)
    return env


def _without_external_qmul_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("TT_RQM_EXTERNAL_QMUL_DIR", None)
    env.pop("TT_RQM_EXTERNAL_QMUL_MANIFEST", None)
    return env


def _load_script(path: str):
    script_path = Path(path)
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.pop("check_environment", None)
    sys.path.insert(0, str(script_path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop("check_environment", None)
        sys.path.pop(0)
    return module
