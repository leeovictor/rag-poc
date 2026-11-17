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


def test_print_queries_escapes_markup(monkeypatch) -> None:
    test_console = Console(record=True, width=100, force_terminal=False)
    monkeypatch.setattr(ui, "console", test_console)

    ui.print_queries(["useState[]", "cleanup timing"])

    output = test_console.export_text()
    assert "Consultas (2)" in output
    assert "useState[]" in output
    assert "cleanup timing" in output


def test_print_sources_adds_subquery_column(monkeypatch, tmp_path) -> None:
    file = tmp_path / "doc.md"
    file.write_text("conteudo", encoding="utf-8")
    chunk = RetrievedChunk(
        document=Document(
            page_content="conteudo",
            metadata={"source": str(file), "file_name": "doc.md"},
        ),
        score=1.0,
        sources=("vetor", "bm25"),
        queries=("q1", "q2"),
    )
    test_console = Console(record=True, width=140, force_terminal=False)
    monkeypatch.setattr(ui, "console", test_console)

    ui.print_sources([chunk], {1}, link_scheme="none")

    output = test_console.export_text()
    assert "Subconsultas" in output
    assert "2" in output


def test_linkify_citations_maps_to_source_links(monkeypatch, tmp_path) -> None:
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

    result = ui.linkify_citations("Veja [1] e [2].", [chunk], "vscode")

    assert "[[1]](vscode://file" in result
    assert "[2]" in result


def test_linkify_citations_ignores_code_blocks(monkeypatch, tmp_path) -> None:
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

    result = ui.linkify_citations("```\n[1]\n```\nVeja [1]", [chunk], "vscode")

    assert "```\n[1]\n```" in result
    assert result.endswith("[[1]](vscode://file" + file.resolve().as_posix() + ")")


def test_linkify_citations_disabled_scheme() -> None:
    assert ui.linkify_citations("Veja [1]", [], "none") == "Veja [1]"


def test_print_gap_is_informative(monkeypatch) -> None:
    test_console = Console(record=True, width=100, force_terminal=False)
    monkeypatch.setattr(ui, "console", test_console)

    ui.print_gap("falta detalhe", ["gap query"])

    output = test_console.export_text()
    assert "Lacuna detectada" in output
    assert "falta detalhe" in output
    assert "1 consulta(s)" in output
