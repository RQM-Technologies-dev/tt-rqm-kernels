from pathlib import Path


def test_h2a_single_core_candidate_source_is_present_and_pinned() -> None:
    root = Path("experimental/tt_metalium_hamiltonian_lowering")
    text = (root / "README.md").read_text(encoding="utf-8")
    assert "dd2849b5bc6b7a5d38a9eafbeba31ef8d530f8d4" in text
    assert "H2B" in text
    assert (root / "src/hamiltonian_lowering_candidate.cpp").is_file()
    assert (root / "src/kernels/compute_hamiltonian_lowering.cpp").is_file()
    assert (root / "src/kernels/reader_hamiltonian_lowering.cpp").is_file()
    assert (root / "src/kernels/writer_hamiltonian_lowering.cpp").is_file()


def test_h2a_compute_source_uses_safe_zero_before_reciprocal() -> None:
    text = Path(
        "experimental/tt_metalium_hamiltonian_lowering/src/kernels/compute_hamiltonian_lowering.cpp"
    ).read_text(encoding="utf-8")
    assert text.index("h2a_select(r, zero_mask, safe_r, 0)") < text.index(
        "h2a_unary(safe_r, inv_r, 1)"
    )
    assert "eqz_tile(0)" in text
    assert "recip_tile<false>(0)" in text
    assert "sin_tile(0)" in text
    assert "cos_tile(0)" in text
