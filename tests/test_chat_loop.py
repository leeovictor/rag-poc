from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import openai
from langchain_core.documents import Document

from rag_poc import cli, ui
from rag_poc.chain import SufficiencyResult
from rag_poc.retriever import RetrievedChunk


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        document=Document(
            page_content="useState lets you add state.",
            metadata={"doc_id": "a", "source": "a.md", "file_name": "a.md"},
        ),
        score=1.0,
        sources=("vetor",),
        queries=("useState",),
    )


class _FakeAnswerChain:
    def __init__(self) -> None:
        self.inputs: list[dict] = []

    def stream(self, payload: dict):
        self.inputs.append(payload)
        return iter(["resposta"])


class _FakeRetriever:
    hybrid_enabled = False
    document_count = 1

    def __init__(self, chunks: list[RetrievedChunk] | None = None) -> None:
        self.calls: list[list[str]] = []
        self._chunks = chunks if chunks is not None else [_chunk()]

    def retrieve_many(self, queries: list[str]) -> list[RetrievedChunk]:
        self.calls.append(list(queries))
        return self._chunks

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        return self.retrieve_many([query])


def _question_provider(questions: list[str]):
    pending = iter(questions)

    def ask() -> str | None:
        return next(pending, None)

    return ask


class _Records:
    def __init__(self) -> None:
        self.answer = _FakeAnswerChain()
        self.retriever = _FakeRetriever()
        self.planner_calls: list[dict] = []
        self.rewritten: list[str] = []
        self.printed_queries: list[list[str]] = []
        self.fallbacks = 0
        self.no_results: list[tuple[str, int]] = []
        self.check_calls: list[dict] = []
        self.gaps: list[tuple[str, list[str]]] = []
        self.sufficiency_fallbacks = 0


def _run_chat(
    monkeypatch,
    settings,
    questions: list[str],
    *,
    plan=None,
    sufficiency=None,
    records: _Records | None = None,
    planner_error: bool = False,
    sufficiency_error: bool = False,
    settings_override=None,
):
    records = records or _Records()

    def default_plan(question: str, history: str) -> list[str]:
        return [question]

    plan_fn = plan or default_plan

    def default_sufficiency(question: str, history: str, chunks: list) -> SufficiencyResult:
        return SufficiencyResult(sufficient=True, missing="", queries=[])

    sufficiency_fn = sufficiency or default_sufficiency

    def fake_check(chain, question, history, chunks, max_gap_queries):
        records.check_calls.append(
            {
                "question": question,
                "history": history,
                "chunks": list(chunks),
                "max_gap_queries": max_gap_queries,
            }
        )
        if sufficiency_error:
            raise openai.APIConnectionError(request=None)  # type: ignore[arg-type]
        return sufficiency_fn(question, history, list(chunks))

    def fail_or_plan(chain, question: str, history: str, max_queries: int) -> list[str]:
        records.planner_calls.append(
            {"question": question, "history": history, "max_queries": max_queries}
        )
        if planner_error:
            raise openai.APIConnectionError(request=None)  # type: ignore[arg-type]
        return plan_fn(question, history)

    monkeypatch.setattr(ui, "ask_question", _question_provider(questions))
    monkeypatch.setattr(ui, "chat_intro", lambda *args, **kwargs: None)
    monkeypatch.setattr(ui, "stream_answer", lambda stream, **kwargs: "resposta")
    monkeypatch.setattr(ui, "print_sources", lambda *args, **kwargs: None)
    monkeypatch.setattr(ui, "print_rewritten_query", records.rewritten.append)
    monkeypatch.setattr(ui, "print_queries", records.printed_queries.append)
    monkeypatch.setattr(
        ui, "print_planner_fallback", lambda: setattr(records, "fallbacks", records.fallbacks + 1)
    )
    monkeypatch.setattr(
        ui,
        "print_no_results",
        lambda question, count: records.no_results.append((question, count)),
    )
    monkeypatch.setattr(
        ui,
        "print_gap",
        lambda missing, queries: records.gaps.append((missing, list(queries))),
    )
    monkeypatch.setattr(
        ui,
        "print_sufficiency_fallback",
        lambda: setattr(
            records,
            "sufficiency_fallbacks",
            records.sufficiency_fallbacks + 1,
        ),
    )

    test_settings = settings_override or replace(settings, chat_memory_turns=2)

    with (
        patch.object(cli.Settings, "from_env", return_value=test_settings),
        patch.object(cli.Retriever, "build", return_value=records.retriever),
        patch.object(cli, "build_answer_chain", return_value=records.answer),
        patch.object(cli, "build_planner_chain", return_value=object()),
        patch.object(cli, "plan_queries", side_effect=fail_or_plan),
        patch.object(cli, "build_sufficiency_chain", return_value=object()),
        patch.object(cli, "check_sufficiency", side_effect=fake_check),
    ):
        cli.chat(k=None, show_chunks=False, debug=False)

    return records


def test_chat_persists_history_and_rewrites_followup(monkeypatch, settings) -> None:
    records = _run_chat(
        monkeypatch,
        settings,
        ["o que é useState?", "e o valor inicial?", "sair"],
        plan=lambda question, history: (
            ["useState initial value"] if question == "e o valor inicial?" else [question]
        ),
    )

    assert [payload["question"] for payload in records.answer.inputs] == [
        "o que é useState?",
        "e o valor inicial?",
    ]
    assert records.answer.inputs[0]["history"] == []
    assert [message.content for message in records.answer.inputs[1]["history"]] == [
        "o que é useState?",
        "resposta",
    ]

    assert len(records.planner_calls) == 2
    assert records.planner_calls[1]["question"] == "e o valor inicial?"
    assert "Usuário: o que é useState?" in records.planner_calls[1]["history"]
    assert records.retriever.calls == [["o que é useState?"], ["useState initial value"]]
    assert records.rewritten == ["useState initial value"]


def test_chat_limpar_resets_history(monkeypatch, settings) -> None:
    records = _run_chat(monkeypatch, settings, ["primeira", "limpar", "segunda", "sair"])

    assert [payload["question"] for payload in records.answer.inputs] == [
        "primeira",
        "segunda",
    ]
    assert records.answer.inputs[1]["history"] == []
    assert records.planner_calls[-1]["history"] == ""


def test_chat_memory_disabled_sends_empty_history(monkeypatch, settings) -> None:
    test_settings = replace(settings, chat_memory_turns=0, fanout_enabled=False)

    records = _run_chat(
        monkeypatch,
        settings,
        ["primeira", "segunda", "sair"],
        settings_override=test_settings,
    )

    assert [payload["history"] for payload in records.answer.inputs] == [[], []]
    assert records.planner_calls == []
    assert records.retriever.calls == [["primeira"], ["segunda"]]


def test_chat_planner_failure_falls_back_to_original(monkeypatch, settings) -> None:
    records = _run_chat(
        monkeypatch,
        settings,
        ["primeira", "segunda", "sair"],
        planner_error=True,
    )

    assert records.retriever.calls == [["primeira"], ["segunda"]]
    assert records.fallbacks == 2
    assert len(records.answer.inputs) == 2


def test_chat_without_chunks_answers_with_general_knowledge(monkeypatch, settings) -> None:
    records = _Records()
    records.retriever = _FakeRetriever(chunks=[])
    records.retriever.document_count = 0
    records = _run_chat(monkeypatch, settings, ["pergunta", "sair"], records=records)

    assert records.no_results == [("pergunta", 0)]
    assert len(records.answer.inputs) == 1
    assert records.answer.inputs[0]["context"] == "(nenhum trecho recuperado)"
    assert records.answer.inputs[0]["question"] == "pergunta"


def test_chat_fanout_passes_multiple_queries(monkeypatch, settings) -> None:
    records = _run_chat(
        monkeypatch,
        settings,
        ["como funciona useMemo e useCallback?", "sair"],
        plan=lambda question, history: [
            "useMemo vs useCallback",
            "useMemo performance",
            "useCallback dependencies",
        ],
    )

    assert records.retriever.calls == [
        ["useMemo vs useCallback", "useMemo performance", "useCallback dependencies"]
    ]
    assert records.printed_queries == [
        ["useMemo vs useCallback", "useMemo performance", "useCallback dependencies"]
    ]
    assert records.answer.inputs[0]["question"] == "como funciona useMemo e useCallback?"


def test_chat_fanout_disabled_condenses_only_with_history(monkeypatch, settings) -> None:
    test_settings = replace(settings, fanout_enabled=False)

    records = _run_chat(
        monkeypatch,
        settings,
        ["primeira", "e a segunda?", "sair"],
        settings_override=test_settings,
        plan=lambda question, history: ["condensed second"] if history else [question],
    )

    assert records.retriever.calls == [["primeira"], ["condensed second"]]
    assert len(records.planner_calls) == 1
    assert records.planner_calls[0]["max_queries"] == 1


def test_chat_gap_filling_adds_queries_and_refetches(monkeypatch, settings) -> None:
    verdicts = iter(
        [
            SufficiencyResult(
                sufficient=False,
                missing="faltou o comportamento de cleanup",
                queries=["useEffect cleanup timing"],
            ),
            SufficiencyResult(sufficient=True, missing="", queries=[]),
        ]
    )

    records = _run_chat(
        monkeypatch,
        settings,
        ["como o useEffect funciona?", "sair"],
        plan=lambda question, history: ["useEffect"],
        sufficiency=lambda question, history, chunks: next(verdicts),
    )

    assert records.retriever.calls == [
        ["useEffect"],
        ["useEffect", "useEffect cleanup timing"],
    ]
    assert records.gaps == [("faltou o comportamento de cleanup", ["useEffect cleanup timing"])]
    assert len(records.check_calls) == 1
    assert records.printed_queries[-1] == [
        "useEffect",
        "useEffect cleanup timing",
    ]


def test_chat_sufficiency_disabled_skips_check(monkeypatch, settings) -> None:
    records = _run_chat(
        monkeypatch,
        settings,
        ["pergunta", "sair"],
        settings_override=replace(settings, sufficiency_max_rounds=0),
    )

    assert records.check_calls == []
    assert records.retriever.calls == [["pergunta"]]


def test_chat_sufficiency_failure_falls_back(monkeypatch, settings) -> None:
    records = _run_chat(
        monkeypatch,
        settings,
        ["primeira", "segunda", "sair"],
        sufficiency_error=True,
    )

    assert records.sufficiency_fallbacks == 2
    assert records.retriever.calls == [["primeira"], ["segunda"]]
    assert len(records.answer.inputs) == 2
