from __future__ import annotations

from pathlib import Path


ROOT = Path("experimental/tt_metalium_su4q_conformance")
CANDIDATE = ROOT / "src/su4q_conformance_candidate.cpp"


def test_candidate_is_one_device_eight_program_device_resident_pipeline() -> None:
    text = CANDIDATE.read_text()
    assert text.count("MeshDevice::create_unit_mesh(0)") == 1
    assert text.count("device->close()") == 1
    assert text.count("stages.push_back") == 8
    assert text.count("EnqueueWriteMeshBuffer") == 2
    assert text.count("EnqueueReadMeshBuffer") == 1
    assert '"program_count", 8 * depth' in text
    assert '"intermediate_d2h_count", 0' in text
    assert '"intermediate_h2d_count", 0' in text
    assert '"performance_eligible", false' in text


def test_stage_order_and_compiler_block_lanes_are_literal() -> None:
    text = CANDIDATE.read_text()
    for marker in (
        "StageKind::Local, 11, 0",
        "StageKind::Local, 15, 1",
        "StageKind::Cartan, 8, 0",
        "StageKind::Cartan, 9, 1",
        "StageKind::Cartan, 10, 2",
        "StageKind::Local, 0, 0",
        "StageKind::Local, 4, 1",
        "StageKind::Phase, 19, 0",
    ):
        assert marker in text


def test_kernel_sources_encode_required_order_sign_and_no_normalization() -> None:
    local = (ROOT / "kernels/compute_local.cpp").read_text()
    cartan = (ROOT / "kernels/compute_cartan.cpp").read_text()
    phase = (ROOT / "kernels/compute_phase.cpp").read_text()
    assert "pair_stride = qubit == 0 ? 1 : 2" in local
    assert "partner = amplitude ^ 3U" in cartan
    assert "positive = amplitude == 1 || amplitude == 2" in cartan
    assert "su4q_accumulate_product(sine, partner_cb, subtract)" in cartan
    assert "su4q_accumulate_product(sine, partner, !imaginary)" in phase
    assert "normaliz" not in local.lower()
    assert "normaliz" not in cartan.lower()
    assert "normaliz" not in phase.lower()


def test_narrow_build_uses_warnings_as_errors_and_only_new_sources() -> None:
    cmake = (ROOT / "CMakeLists.txt").read_text()
    assert "tt_rqm_metalium_su4q_conformance" in cmake
    assert "-Wall -Wextra -Werror" in cmake
    for marker in (
        "reader_stage.cpp",
        "compute_local.cpp",
        "compute_cartan.cpp",
        "compute_phase.cpp",
        "writer_state.cpp",
    ):
        assert marker in cmake
