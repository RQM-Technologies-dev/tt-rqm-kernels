from pathlib import Path


def test_h2a_scaffold_is_design_only_and_pinned() -> None:
    root = Path("experimental/tt_metalium_hamiltonian_lowering")
    text = (root / "README.md").read_text(encoding="utf-8")
    assert "does not contain an executable TT-Metal candidate" in text
    assert "dd2849b5bc6b7a5d38a9eafbeba31ef8d530f8d4" in text
    assert "available with restrictions" in text
    assert "requires approximation or composed implementation" in text
    assert "not yet verified" in text
    assert "H2B" in text
    assert not list(root.rglob("*.cpp"))
