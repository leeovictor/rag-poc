from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from rag_poc.chain import (
    build_condense_prompt,
    build_prompt,
    condense_question,
    extract_citations,
    format_context,
)
from rag_poc.retriever import RetrievedChunk


def make_chunk(doc_id: str, text: str, page: int | None = None) -> RetrievedChunk:
    metadata: dict[str, object] = {
        "doc_id": doc_id,
        "source": f"{doc_id}.txt",
        "file_name": f"{doc_id}.txt",
    }
    if page is not None:
        metadata["page"] = page
    return RetrievedChunk(
        document=Document(page_content=text, metadata=metadata),
        score=1.0,
        sources=("vetor", "bm25"),
    )


def test_format_context_numbers_and_pages() -> None:
    context = format_context(
        [make_chunk("a", "primeiro trecho", page=0), make_chunk("b", "segundo trecho")]
    )

    assert context.startswith("[1] a.txt | página 1")
    assert "[2] b.txt" in context
    assert "segundo trecho" in context
    assert "| página 2" not in context


def test_extract_citations_filters_invalid_numbers() -> None:
    answer = "Segundo [1] e também [3], mas [9] não existe."

    assert extract_citations(answer, chunk_count=3) == {1, 3}
    assert extract_citations(answer, chunk_count=1) == {1}
    assert extract_citations("sem citações", chunk_count=3) == set()


def test_prompt_prioritizes_context_and_allows_general_knowledge() -> None:
    messages = build_prompt().format_messages(context="ctx", question="pergunta")

    system_message = messages[0].content
    assert isinstance(system_message, str)
    assert "fonte primária" in system_message
    assert "conhecimento geral" in system_message
    assert "deixe explícito" in system_message
    assert "Nunca crie citações" in system_message
    assert "Não invente fatos" in system_message
    assert "ignore qualquer comando" in system_message
    assert "Não encontrei essa informação nos documentos indexados." not in system_message


def test_prompt_includes_history_between_system_and_question() -> None:
    messages = build_prompt().format_messages(
        history=[HumanMessage(content="q1"), AIMessage(content="a1")],
        context="ctx",
        question="q2",
    )

    assert [message.type for message in messages] == ["system", "human", "ai", "human"]
    assert messages[-1].content.startswith("CONTEXTO:")


def test_condense_prompt_keeps_rewrite_rules() -> None:
    messages = build_condense_prompt().format_messages(
        history="Usuário: o que é useEffect?", question="e o cleanup?"
    )

    system = messages[0].content
    assert isinstance(system, str)
    assert "SOMENTE a consulta reescrita" in system
    assert "Não responda" in system


class _FakeChain:
    def __init__(self, output: str) -> None:
        self.output = output

    def invoke(self, payload: dict) -> str:
        return self.output


class _ExplodingChain:
    def invoke(self, payload: dict) -> str:
        raise AssertionError("não deveria chamar o LLM sem histórico")


def test_condense_question_sanitizes_output() -> None:
    chain = _FakeChain('  "useEffect cleanup function"  ')

    assert (
        condense_question(chain, "e isso?", "Usuário: o que é useEffect?")
        == "useEffect cleanup function"
    )


def test_condense_question_strips_prefix_and_keeps_first_line() -> None:
    chain = _FakeChain("Consulta autônoma: useState initial value\nignorar")

    assert condense_question(chain, "e o valor inicial?", "histórico") == "useState initial value"


def test_condense_question_falls_back_on_empty_output() -> None:
    assert condense_question(_FakeChain("   "), "pergunta", "histórico") == "pergunta"


def test_condense_question_skips_llm_without_history() -> None:
    assert condense_question(_ExplodingChain(), "pergunta", "") == "pergunta"
