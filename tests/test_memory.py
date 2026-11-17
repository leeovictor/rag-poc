from __future__ import annotations

import pytest

from rag_poc.memory import ConversationMemory


def test_memory_keeps_only_last_turns() -> None:
    memory = ConversationMemory(max_turns=2)
    for index in range(3):
        memory.add_turn(f"q{index}", f"a{index}")

    assert [message.content for message in memory.messages()] == [
        "q1",
        "a1",
        "q2",
        "a2",
    ]
    assert memory.turn_count == 2


def test_memory_disabled_does_not_store() -> None:
    memory = ConversationMemory(max_turns=0)
    memory.add_turn("q", "a")

    assert memory.enabled is False
    assert memory.messages() == []
    assert memory.turn_count == 0


def test_memory_clear() -> None:
    memory = ConversationMemory(max_turns=3)
    memory.add_turn("q", "a")

    memory.clear()

    assert memory.messages() == []


def test_memory_as_text_has_roles() -> None:
    memory = ConversationMemory(max_turns=2)
    memory.add_turn("o que é useState?", "É um hook de estado.")

    text = memory.as_text()

    assert "Usuário: o que é useState?" in text
    assert "Assistente: É um hook de estado." in text


def test_memory_as_text_truncates() -> None:
    memory = ConversationMemory(max_turns=2)
    memory.add_turn("q" * 100, "a" * 100)

    assert len(memory.as_text(max_chars=20)) == 20


def test_memory_rejects_negative_turns() -> None:
    with pytest.raises(ValueError):
        ConversationMemory(max_turns=-1)
