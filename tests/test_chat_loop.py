from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

from langchain_core.documents import Document

from rag_poc import cli, ui
from rag_poc.retriever import RetrievedChunk


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        document=Document(
            page_content="useState lets you add state.",
            metadata={"doc_id": "a", "source": "a.md", "file_name": "a.md"},
        ),
        score=1.0,
        sources=("vetor",),
    )


class _FakeAnswerChain:
    def __init__(self) -> None:
        self.inputs: list[dict] = []

    def stream(self, payload: dict):
        self.inputs.append(payload)
        return iter(["resposta"])


class _FakeCondenseChain:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def invoke(self, payload: dict) -> str:
        self.calls.append(payload)
        return "useState initial value"


class _FakeRetriever:
    hybrid_enabled = False
    document_count = 1

    def __init__(self) -> None:
        self.queries: list[str] = []

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        self.queries.append(query)
        return [_chunk()]


def _question_provider(questions: list[str]):
    pending = iter(questions)

    def ask() -> str | None:
        return next(pending, None)

    return ask


def _run_chat(monkeypatch, settings, questions: list[str]):
    fake_answer = _FakeAnswerChain()
    fake_condense = _FakeCondenseChain()
    fake_retriever = _FakeRetriever()
    rewritten: list[str] = []

    monkeypatch.setattr(ui, "ask_question", _question_provider(questions))
    monkeypatch.setattr(ui, "chat_intro", lambda *args, **kwargs: None)
    monkeypatch.setattr(ui, "stream_answer", lambda stream: "resposta")
    monkeypatch.setattr(ui, "print_sources", lambda *args, **kwargs: None)
    monkeypatch.setattr(ui, "print_rewritten_query", rewritten.append)
    monkeypatch.setattr(ui, "print_condense_fallback", lambda: None)

    test_settings = replace(settings, chat_memory_turns=2)

    with (
        patch.object(cli.Settings, "from_env", return_value=test_settings),
        patch.object(cli.Retriever, "build", return_value=fake_retriever),
        patch.object(cli, "build_answer_chain", return_value=fake_answer),
        patch.object(cli, "build_condense_chain", return_value=fake_condense),
    ):
        cli.chat(k=None, show_chunks=False, debug=False)

    return fake_answer, fake_condense, fake_retriever, rewritten


def test_chat_persists_history_and_rewrites_followup(monkeypatch, settings) -> None:
    fake_answer, fake_condense, fake_retriever, rewritten = _run_chat(
        monkeypatch, settings, ["o que é useState?", "e o valor inicial?", "sair"]
    )

    assert [payload["question"] for payload in fake_answer.inputs] == [
        "o que é useState?",
        "e o valor inicial?",
    ]
    assert fake_answer.inputs[0]["history"] == []
    assert [message.content for message in fake_answer.inputs[1]["history"]] == [
        "o que é useState?",
        "resposta",
    ]

    assert len(fake_condense.calls) == 1
    assert fake_condense.calls[0]["question"] == "e o valor inicial?"
    assert "Usuário: o que é useState?" in fake_condense.calls[0]["history"]
    assert fake_retriever.queries == ["o que é useState?", "useState initial value"]
    assert rewritten == ["useState initial value"]


def test_chat_limpar_resets_history(monkeypatch, settings) -> None:
    fake_answer, fake_condense, _retriever, _rewritten = _run_chat(
        monkeypatch, settings, ["primeira", "limpar", "segunda", "sair"]
    )

    assert [payload["question"] for payload in fake_answer.inputs] == [
        "primeira",
        "segunda",
    ]
    assert fake_answer.inputs[1]["history"] == []
    assert fake_condense.calls == []


def test_chat_memory_disabled_sends_empty_history(monkeypatch, settings) -> None:
    test_settings = replace(settings, chat_memory_turns=0)

    fake_answer = _FakeAnswerChain()
    fake_condense = _FakeCondenseChain()

    monkeypatch.setattr(ui, "ask_question", _question_provider(["primeira", "segunda", "sair"]))
    monkeypatch.setattr(ui, "chat_intro", lambda *args, **kwargs: None)
    monkeypatch.setattr(ui, "stream_answer", lambda stream: "resposta")
    monkeypatch.setattr(ui, "print_sources", lambda *args, **kwargs: None)
    monkeypatch.setattr(ui, "print_rewritten_query", lambda *args: None)
    monkeypatch.setattr(ui, "print_condense_fallback", lambda: None)

    with (
        patch.object(cli.Settings, "from_env", return_value=test_settings),
        patch.object(cli.Retriever, "build", return_value=_FakeRetriever()),
        patch.object(cli, "build_answer_chain", return_value=fake_answer),
        patch.object(cli, "build_condense_chain", return_value=fake_condense),
    ):
        cli.chat(k=None, show_chunks=False, debug=False)

    assert [payload["history"] for payload in fake_answer.inputs] == [[], []]
    assert fake_condense.calls == []


def test_chat_condense_failure_falls_back_to_original(monkeypatch, settings) -> None:
    import openai

    class _FailingCondense:
        def invoke(self, payload: dict) -> str:
            raise openai.APIConnectionError(request=None)  # type: ignore[arg-type]

    fake_answer = _FakeAnswerChain()
    fake_retriever = _FakeRetriever()
    fallbacks: list[bool] = []

    monkeypatch.setattr(ui, "ask_question", _question_provider(["primeira", "segunda", "sair"]))
    monkeypatch.setattr(ui, "chat_intro", lambda *args, **kwargs: None)
    monkeypatch.setattr(ui, "stream_answer", lambda stream: "resposta")
    monkeypatch.setattr(ui, "print_sources", lambda *args, **kwargs: None)
    monkeypatch.setattr(ui, "print_rewritten_query", lambda *args: None)
    monkeypatch.setattr(ui, "print_condense_fallback", lambda: fallbacks.append(True))

    with (
        patch.object(cli.Settings, "from_env", return_value=settings),
        patch.object(cli.Retriever, "build", return_value=fake_retriever),
        patch.object(cli, "build_answer_chain", return_value=fake_answer),
        patch.object(cli, "build_condense_chain", return_value=_FailingCondense()),
    ):
        cli.chat(k=None, show_chunks=False, debug=False)

    assert fake_retriever.queries == ["primeira", "segunda"]
    assert fallbacks == [True]
    assert len(fake_answer.inputs) == 2


class _EmptyRetriever:
    hybrid_enabled = False
    document_count = 0

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        return []


def test_chat_without_chunks_answers_with_general_knowledge(monkeypatch, settings) -> None:
    fake_answer = _FakeAnswerChain()
    fake_condense = _FakeCondenseChain()
    warnings: list[tuple[str, int]] = []

    monkeypatch.setattr(ui, "ask_question", _question_provider(["pergunta", "sair"]))
    monkeypatch.setattr(ui, "chat_intro", lambda *args, **kwargs: None)
    monkeypatch.setattr(ui, "stream_answer", lambda stream: "resposta")
    monkeypatch.setattr(ui, "print_sources", lambda *args, **kwargs: None)
    monkeypatch.setattr(ui, "print_rewritten_query", lambda *args: None)
    monkeypatch.setattr(ui, "print_condense_fallback", lambda: None)
    monkeypatch.setattr(
        ui, "print_no_results", lambda question, count: warnings.append((question, count))
    )

    with (
        patch.object(cli.Settings, "from_env", return_value=settings),
        patch.object(cli.Retriever, "build", return_value=_EmptyRetriever()),
        patch.object(cli, "build_answer_chain", return_value=fake_answer),
        patch.object(cli, "build_condense_chain", return_value=fake_condense),
    ):
        cli.chat(k=None, show_chunks=False, debug=False)

    assert warnings == [("pergunta", 0)]
    assert len(fake_answer.inputs) == 1
    assert fake_answer.inputs[0]["context"] == "(nenhum trecho recuperado)"
    assert fake_answer.inputs[0]["question"] == "pergunta"
