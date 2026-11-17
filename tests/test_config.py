from __future__ import annotations

import pytest

from rag_poc.config import Settings, normalize_model_id
from rag_poc.errors import SettingsError


def test_normalize_model_id_adds_provider_prefix() -> None:
    assert normalize_model_id("gpt-4o-mini") == ("openai/gpt-4o-mini", True)
    assert normalize_model_id("text-embedding-3-small") == (
        "openai/text-embedding-3-small",
        True,
    )
    assert normalize_model_id("anthropic/claude-3.5-sonnet") == (
        "anthropic/claude-3.5-sonnet",
        False,
    )


def test_hybrid_weights_are_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HYBRID_WEIGHTS", "1,3")
    settings = Settings.from_env()
    assert settings.hybrid_weights == pytest.approx((0.25, 0.75))


def test_invalid_hybrid_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HYBRID_WEIGHTS", "0,0")
    with pytest.raises(SettingsError):
        Settings.from_env()


def test_chunk_overlap_must_be_smaller_than_chunk_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHUNK_SIZE", "100")
    monkeypatch.setenv("CHUNK_OVERLAP", "100")
    with pytest.raises(SettingsError):
        Settings.from_env()


def test_require_api_key_raises_when_missing(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    # env_file inexistente isola o teste de um .env real na raiz do projeto.
    settings = Settings.from_env(env_file=tmp_path / "none.env")
    with pytest.raises(SettingsError, match="OPENROUTER_API_KEY"):
        settings.require_api_key()


def test_models_without_prefix_generate_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
    settings = Settings.from_env()
    assert settings.llm_model == "openai/gpt-4o-mini"
    assert settings.embedding_model == "openai/text-embedding-3-small"
    assert any("LLM_MODEL" in warning for warning in settings.warnings)
    assert any("EMBEDDING_MODEL" in warning for warning in settings.warnings)


def test_chat_memory_turns_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHAT_MEMORY_TURNS", "0")
    assert Settings.from_env().chat_memory_turns == 0


def test_chat_memory_turns_rejects_negative(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHAT_MEMORY_TURNS", "-1")
    with pytest.raises(SettingsError):
        Settings.from_env()


def test_citation_link_scheme_defaults_to_vscode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CITATION_LINK_SCHEME", raising=False)
    assert Settings.from_env().citation_link_scheme == "vscode"


def test_citation_link_scheme_can_be_overridden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CITATION_LINK_SCHEME", "file")
    assert Settings.from_env().citation_link_scheme == "file"


def test_citation_link_scheme_rejects_invalid_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CITATION_LINK_SCHEME", "emacs")
    with pytest.raises(SettingsError):
        Settings.from_env()
