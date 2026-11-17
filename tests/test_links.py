from __future__ import annotations

from rag_poc.links import build_source_link


def test_none_scheme_returns_none(tmp_path) -> None:
    file = tmp_path / "a.md"
    file.write_text("x", encoding="utf-8")

    assert build_source_link(str(file), "none") is None


def test_missing_file_returns_none(tmp_path) -> None:
    assert build_source_link(str(tmp_path / "nao-existe.md"), "vscode") is None


def test_vscode_link(tmp_path) -> None:
    file = tmp_path / "a.md"
    file.write_text("x", encoding="utf-8")

    link = build_source_link(str(file), "vscode")

    assert link == f"vscode://file{file.resolve().as_posix()}"


def test_vscode_link_encodes_spaces(tmp_path) -> None:
    file = tmp_path / "my doc.md"
    file.write_text("x", encoding="utf-8")

    link = build_source_link(str(file), "vscode")

    assert link is not None
    assert "my%20doc.md" in link


def test_file_link_appends_pdf_page(tmp_path) -> None:
    file = tmp_path / "doc.pdf"
    file.write_bytes(b"%PDF-1.4")

    link = build_source_link(str(file), "file", page=2)

    assert link == file.resolve().as_uri() + "#page=3"


def test_invalid_scheme_returns_none(tmp_path) -> None:
    file = tmp_path / "a.md"
    file.write_text("x", encoding="utf-8")

    assert build_source_link(str(file), "editor-x") is None
