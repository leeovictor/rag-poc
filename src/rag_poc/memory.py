"""Memória de conversa em processo, com janela de turnos."""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

DEFAULT_MAX_CHARS = 4000


@dataclass
class ConversationMemory:
    """Histórico de mensagens em memória, limitado aos últimos `max_turns` turnos.

    `max_turns=0` desativa a memória. O histórico não é persistido em disco.
    """

    max_turns: int = 6
    _history: InMemoryChatMessageHistory = field(default_factory=InMemoryChatMessageHistory)

    def __post_init__(self) -> None:
        if self.max_turns < 0:
            raise ValueError("max_turns não pode ser negativo.")

    @property
    def enabled(self) -> bool:
        return self.max_turns > 0

    @property
    def turn_count(self) -> int:
        return len(self.messages()) // 2

    def messages(self) -> list[BaseMessage]:
        """Mensagens dentro da janela, na ordem original."""
        if not self.enabled:
            return []
        return self._history.messages[-(self.max_turns * 2) :]

    def add_turn(self, question: str, answer: str) -> None:
        if not self.enabled:
            return
        self._history.add_message(HumanMessage(content=question))
        self._history.add_message(AIMessage(content=answer))
        self._trim()

    def clear(self) -> None:
        self._history.clear()

    def as_text(self, max_chars: int = DEFAULT_MAX_CHARS) -> str:
        """Histórico legível para o prompt de reformulação de consulta."""
        lines: list[str] = []
        for message in self.messages():
            role = "Usuário" if isinstance(message, HumanMessage) else "Assistente"
            content = message.content if isinstance(message.content, str) else str(message.content)
            lines.append(f"{role}: {content}")
        text = "\n".join(lines)
        return text[-max_chars:] if len(text) > max_chars else text

    def _trim(self) -> None:
        limit = self.max_turns * 2
        if len(self._history.messages) <= limit:
            return
        kept = self._history.messages[-limit:]
        self._history.clear()
        for message in kept:
            self._history.add_message(message)
