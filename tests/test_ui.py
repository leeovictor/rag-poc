from __future__ import annotations

from langchain_core.documents import Document
from rich.console import Console

from rag_poc import ui
from rag_poc.retriever import RetrievedChunk


def test_print_no_results_is_informative_and_escapes_markup(monkeypatch) -> None:
    test_console = Console(record=True, width=100, force_terminal=False)
    monkeypatch.setattr(ui, "console", test_console)

    ui.print_no_results("useState[]?", 42)

    output = test_console.export_text()
    assert "Não encontrei trechos relevantes" in output
    assert "useState[]?" in output
    assert "42 chunk(s)" in output
    assert "conhecimento geral" in output


def test_print_sources_renders_filename_with_link(monkeypatch, tmp_path) -> None:
    file = tmp_path / "doc.md"
    file.write_text("conteudo", encoding="utf-8")
    chunk = RetrievedChunk(
        document=Document(
            page_content="conteudo",
            metadata={"source": str(file), "file_name": "doc.md"},
        ),
        score=1.0,
        sources=("vetor",),
    )
    test_console = Console(record=True, width=120, force_terminal=False)
    monkeypatch.setattr(ui, "console", test_console)

    ui.print_sources([chunk], {1}, link_scheme="vscode")

    output = test_console.export_text()
    assert "doc.md" in output
    assert "vetor" in output
