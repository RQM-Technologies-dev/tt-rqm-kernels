from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path("experimental/tt_metalium_hamiltonian_evolution")
CANDIDATE = ROOT / "src/hamiltonian_evolution_candidate.cpp"
PROTECTED_HASHES = {
    "experimental/tt_metalium_su2_compose/kernels/su2_fused_reader.cpp": "b8bb98fe6be44083cac654382397b2740eaef88af6ed4759e429391e8398b909",
    "experimental/tt_metalium_su2_compose/kernels/su2_fused_compute.cpp": "a721d4caeec48c73293c010b4395891e0196b9993b169f02ef5722e7fe5d0b58",
    "experimental/tt_metalium_su2_compose/kernels/su2_fused_writer.cpp": "38bd138019607e0befc6ba3f02a918be725728936958f794c0c5ca11c0662819",
    "experimental/tt_metalium_su2_compose/kernels/su2_compute_common.h": "8fb8d0634cfa80c158284ed8b08b1821de8e66698925ae1c317845cc555b20ad",
    "experimental/tt_metalium_su2_compose/kernels/su2_sfpu.h": "d2e2b26a988377588f1d49d10a4c7a664c5520cc24938471b8a46cfe21c58542",
    "experimental/tt_metalium_hamiltonian_lowering_compensated/kernels/compute_hamiltonian_lowering_compensated.cpp": "c9b45e774924a8d67079bd330d39238ad3d73f8f163830274c579dbd756f1592",
    "experimental/tt_metalium_hamiltonian_lowering_compensated/kernels/h2a_compensated_sfpu.h": "01476a689ff6dd84ce4e96ff7e8f0307522d595a4fad4bd1eb215277ac1b9330",
}


def test_h2b_candidate_has_one_device_two_program_resident_architecture() -> None:
    text = CANDIDATE.read_text(encoding="utf-8")
    assert text.count("MeshDevice::create_unit_mesh(0)") == 1
    assert text.count("device->close()") == 1
    assert "build_h2a_program(device, input, intermediate" in text
    assert "build_h1_program(device, intermediate, final_output" in text
    assert text.count("EnqueueMeshWorkload") == 2
    assert text.count("EnqueueWriteMeshBuffer") == 1
    assert text.count("EnqueueReadMeshBuffer") == 1
    assert "EnqueueReadMeshBuffer(queue, packed_final, final_output" in text
    assert 'intermediate_d2h_count", 0' in text
    assert 'intermediate_h2d_count", 0' in text
    assert 'host_round_trip_count", 0' in text


def test_h2a_step_major_reader_writer_use_required_page_mapping() -> None:
    for relative in (
        "kernels/reader_hamiltonian_lowering_step_major.cpp",
        "kernels/writer_hamiltonian_lowering_step_major.cpp",
    ):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "step = work_item / component_tiles" in text
        assert "batch_tile = work_item % component_tiles" in text
        assert "(step * 6 + lane) * component_tiles + batch_tile" in text


def test_h2b_build_reuses_exact_h2a_and_h1_sources() -> None:
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    for marker in (
        "compute_hamiltonian_lowering_compensated.cpp",
        "su2_fused_reader.cpp",
        "su2_fused_compute.cpp",
        "su2_fused_writer.cpp",
    ):
        assert marker in cmake
    for relative, expected in PROTECTED_HASHES.items():
        actual = hashlib.sha256(Path(relative).read_bytes()).hexdigest()
        assert actual == expected


def test_h2b_metadata_describes_two_program_pipeline_not_one_fused_kernel() -> None:
    text = CANDIDATE.read_text(encoding="utf-8")
    assert '"program_count", 2' in text
    assert '"intermediate_storage", "device_dram"' in text
    assert '"device_resident_intermediate", true' in text
    assert '"composition_order", "K-1 ... 0"' in text
    assert '"automatic_normalization", false' in text
    assert "single_fused_kernel" not in text
