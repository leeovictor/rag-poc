from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from rag_poc.chain import (
    SufficiencyResult,
    build_planner_prompt,
    build_prompt,
    build_sufficiency_prompt,
    check_sufficiency,
    extract_citations,
    format_context,
    merge_queries,
    parse_queries,
    parse_sufficiency,
    plan_queries,
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


def test_planner_prompt_includes_limit_language_and_json_rule() -> None:
    messages = build_planner_prompt(3, "en").format_messages(
        history="Usuário: o que é useEffect?", question="e o cleanup?"
    )

    system = messages[0].content
    assert isinstance(system, str)
    assert "no máximo 3 consultas" in system
    assert "inglês" in system
    assert "JSON" in system
    assert "Não responda" in system
    assert messages[1].content.startswith("HISTÓRICO:")


def test_parse_queries_json() -> None:
    assert parse_queries('{"queries": ["a", "b"]}', 3) == ["a", "b"]


def test_parse_queries_fenced_json() -> None:
    raw = '```json\n{"queries": ["useState", "cleanup timing"]}\n```'

    assert parse_queries(raw, 3) == ["useState", "cleanup timing"]


def test_parse_queries_plain_lines_fallback() -> None:
    assert parse_queries("- useState\n- cleanup timing", 3) == [
        "useState",
        "cleanup timing",
    ]


def test_parse_queries_dedupes_and_clamps() -> None:
    raw = '{"queries": ["A", "a", "b", "c", "d"]}'

    assert parse_queries(raw, 3) == ["A", "b", "c"]


def test_parse_queries_cleans_bullets_and_numbers() -> None:
    raw = '{"queries": ["- useState initial value", "2) cleanup timing"]}'

    assert parse_queries(raw, 3) == ["useState initial value", "cleanup timing"]


def test_parse_queries_invalid_returns_empty() -> None:
    assert parse_queries("desculpe, não sei responder.", 3) == []


class _FakeChain:
    def __init__(self, output: str) -> None:
        self.output = output
        self.payloads: list[dict] = []

    def invoke(self, payload: dict) -> str:
        self.payloads.append(payload)
        return self.output


def test_plan_queries_invokes_chain_with_history() -> None:
    chain = _FakeChain('{"queries": ["When does useEffect cleanup run?"]}')

    queries = plan_queries(chain, "e quando ele roda?", "Usuário: o que é useEffect?", 3)

    assert queries == ["When does useEffect cleanup run?"]
    assert chain.payloads[0]["question"] == "e quando ele roda?"
    assert "Usuário: o que é useEffect?" in chain.payloads[0]["history"]


def test_plan_queries_falls_back_to_original_on_invalid_output() -> None:
    assert plan_queries(_FakeChain("desculpe"), "pergunta", "histórico", 3) == ["pergunta"]


def test_sufficiency_prompt_includes_limit_language_and_json_rule() -> None:
    messages = build_sufficiency_prompt(2, "en").format_messages(
        history="(sem histórico)",
        question="pergunta",
        context="[1] trecho",
    )

    system = messages[0].content
    assert isinstance(system, str)
    assert "no máximo 2 consultas" in system
    assert "inglês" in system
    assert "JSON" in system
    assert "Não responda" in system


def test_parse_sufficiency_sufficient() -> None:
    result = parse_sufficiency({"sufficient": True, "missing": "", "queries": []}.__repr__(), 2)
    # repr não é JSON; formato inválido deve ser tratado como suficiente
    assert result == SufficiencyResult(sufficient=True, missing="", queries=[])


def test_parse_sufficiency_false_with_queries() -> None:
    raw = (
        '{"sufficient": false, "missing": "como o cleanup roda", '
        '"queries": ["cleanup timing", "cleanup dependencies"]}'
    )

    result = parse_sufficiency(raw, 2)

    assert result.sufficient is False
    assert result.missing == "como o cleanup roda"
    assert result.queries == ["cleanup timing", "cleanup dependencies"]


def test_parse_sufficiency_caps_and_dedupes_queries() -> None:
    raw = '{"sufficient": false, "queries": ["a", "A", "b", "c"]}'

    assert parse_sufficiency(raw, 2).queries == ["a", "b"]


def test_parse_sufficiency_invalid_is_treated_as_sufficient() -> None:
    result = parse_sufficiency("desculpe, não consigo avaliar", 2)

    assert result.sufficient is True
    assert result.queries == []


def test_check_sufficiency_passes_context_and_history() -> None:
    chain = _FakeChain('{"sufficient": true, "missing": "", "queries": []}')

    result = check_sufficiency(
        chain,
        question="e quando roda?",
        history_text="Usuário: o que é useEffect?",
        chunks=[make_chunk("a", "useEffect cleanup runs after render")],
        max_gap_queries=2,
    )

    payload = chain.payloads[0]
    assert payload["question"] == "e quando roda?"
    assert "Usuário: o que é useEffect?" in payload["history"]
    assert "[1] a.txt" in payload["context"]
    assert result.sufficient is True


def test_merge_queries_preserves_order_and_dedupes() -> None:
    assert merge_queries(["A", "b"], ["a", "C"]) == ["A", "b", "C"]
