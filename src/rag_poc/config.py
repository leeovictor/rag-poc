"""Carregamento e validação da configuração (arquivo .env + ambiente)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from .errors import SettingsError
from .links import VALID_SCHEMES

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

_TRUE_VALUES = {"1", "true", "t", "yes", "y", "on"}
_FALSE_VALUES = {"0", "false", "f", "no", "n", "off"}


def _string(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def _int(name: str, default: int, *, minimum: int | None = None) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip())
    except ValueError as exc:
        raise SettingsError(f"{name} deve ser um inteiro (recebido: {raw!r}).") from exc
    if minimum is not None and value < minimum:
        raise SettingsError(f"{name} deve ser >= {minimum} (recebido: {value}).")
    return value


def _float(
    name: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw.strip())
    except ValueError as exc:
        raise SettingsError(f"{name} deve ser um número (recebido: {raw!r}).") from exc
    if minimum is not None and value < minimum:
        raise SettingsError(f"{name} deve ser >= {minimum} (recebido: {value}).")
    if maximum is not None and value > maximum:
        raise SettingsError(f"{name} deve ser <= {maximum} (recebido: {value}).")
    return value


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    value = raw.strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    raise SettingsError(f"{name} deve ser true/false (recebido: {raw!r}).")


def _optional_int(name: str, *, minimum: int = 1) -> int | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw.strip())
    except ValueError as exc:
        raise SettingsError(f"{name} deve ser um inteiro (recebido: {raw!r}).") from exc
    if value < minimum:
        raise SettingsError(f"{name} deve ser >= {minimum} (recebido: {value}).")
    return value


def _weights(name: str) -> tuple[float, float]:
    raw = os.getenv(name) or "0.5,0.5"
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 2:
        raise SettingsError(
            f"{name} deve conter dois valores separados por vírgula (ex.: 0.5,0.5)."
        )
    try:
        weights = (float(parts[0]), float(parts[1]))
    except ValueError as exc:
        raise SettingsError(f"{name} deve conter números (recebido: {raw!r}).") from exc
    if any(weight < 0 for weight in weights) or sum(weights) <= 0:
        raise SettingsError(f"{name} deve conter pesos >= 0 com soma > 0.")
    total = sum(weights)
    return (weights[0] / total, weights[1] / total)


def normalize_model_id(model: str) -> tuple[str, bool]:
    """Garante o prefixo de provider exigido pelo OpenRouter.

    Retorna ``(model_id_normalizado, foi_alterado)``.
    """
    if "/" in model:
        return model, False
    return f"openai/{model}", True


@dataclass(frozen=True, slots=True)
class Settings:
    openrouter_api_key: str
    openrouter_base_url: str
    llm_model: str
    embedding_model: str
    embedding_dimensions: int | None
    embedding_batch_size: int
    temperature: float
    faiss_index_dir: Path
    chunk_size: int
    chunk_overlap: int
    retrieval_k: int
    hybrid_search: bool
    hybrid_weights: tuple[float, float]
    app_url: str
    app_name: str
    citation_link_scheme: str = "vscode"
    fanout_enabled: bool = True
    fanout_max_queries: int = 3
    fanout_candidates_per_query: int = 8
    fanout_query_language: str = "en"
    sufficiency_max_rounds: int = 1
    sufficiency_max_gap_queries: int = 2
    chat_memory_turns: int = 6
    warnings: tuple[str, ...] = ()

    @classmethod
    def from_env(cls, env_file: str | Path | None = None) -> Settings:
        if env_file is not None:
            load_dotenv(dotenv_path=env_file)
        else:
            load_dotenv()

        warnings: list[str] = []

        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if api_key and not api_key.startswith("sk-or-"):
            warnings.append(
                "OPENROUTER_API_KEY não parece uma chave do OpenRouter (esperado prefixo 'sk-or-')."
            )

        llm_model, llm_changed = normalize_model_id(_string("LLM_MODEL", "openai/gpt-4o-mini"))
        if llm_changed:
            warnings.append(
                f"LLM_MODEL sem prefixo de provider; usando '{llm_model}' "
                "(no OpenRouter o id precisa do prefixo, ex.: openai/gpt-4o-mini)."
            )

        embedding_model, embed_changed = normalize_model_id(
            _string("EMBEDDING_MODEL", "openai/text-embedding-3-small")
        )
        if embed_changed:
            warnings.append(f"EMBEDDING_MODEL sem prefixo de provider; usando '{embedding_model}'.")

        chunk_size = _int("CHUNK_SIZE", 1000, minimum=1)
        chunk_overlap = _int("CHUNK_OVERLAP", 200, minimum=0)
        if chunk_overlap >= chunk_size:
            raise SettingsError(
                "CHUNK_OVERLAP deve ser menor que CHUNK_SIZE "
                f"(atual: {chunk_overlap} >= {chunk_size})."
            )

        citation_link_scheme = _string("CITATION_LINK_SCHEME", "vscode").lower()
        if citation_link_scheme not in VALID_SCHEMES:
            permitidos = ", ".join(sorted(VALID_SCHEMES))
            raise SettingsError(
                "CITATION_LINK_SCHEME deve ser um de: "
                f"{permitidos} (recebido: {citation_link_scheme!r})."
            )

        fanout_max_queries = _int("FANOUT_MAX_QUERIES", 3, minimum=1)
        if fanout_max_queries > 6:
            raise SettingsError(
                f"FANOUT_MAX_QUERIES deve ser <= 6 (recebido: {fanout_max_queries})."
            )

        fanout_query_language = _string("FANOUT_QUERY_LANGUAGE", "en").lower()
        if fanout_query_language not in {"en", "auto"}:
            raise SettingsError(
                "FANOUT_QUERY_LANGUAGE deve ser 'en' ou 'auto' "
                f"(recebido: {fanout_query_language!r})."
            )

        return cls(
            openrouter_api_key=api_key,
            openrouter_base_url=_string("OPENROUTER_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
            llm_model=llm_model,
            embedding_model=embedding_model,
            embedding_dimensions=_optional_int("EMBEDDING_DIMENSIONS"),
            embedding_batch_size=_int("EMBEDDING_BATCH_SIZE", 64, minimum=1),
            temperature=_float("TEMPERATURE", 0.0, minimum=0.0, maximum=2.0),
            faiss_index_dir=Path(_string("FAISS_INDEX_DIR", "./faiss_index")).expanduser(),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            retrieval_k=_int("RETRIEVAL_K", 4, minimum=1),
            hybrid_search=_bool("HYBRID_SEARCH", True),
            hybrid_weights=_weights("HYBRID_WEIGHTS"),
            app_url=_string("APP_URL", "http://localhost"),
            app_name=_string("APP_NAME", "rag-poc"),
            citation_link_scheme=citation_link_scheme,
            fanout_enabled=_bool("FANOUT_ENABLED", True),
            fanout_max_queries=fanout_max_queries,
            fanout_candidates_per_query=_int("FANOUT_CANDIDATES_PER_QUERY", 8, minimum=1),
            fanout_query_language=fanout_query_language,
            sufficiency_max_rounds=_int("SUFFICIENCY_MAX_ROUNDS", 1, minimum=0),
            sufficiency_max_gap_queries=_int("SUFFICIENCY_MAX_GAP_QUERIES", 2, minimum=1),
            chat_memory_turns=_int("CHAT_MEMORY_TURNS", 6, minimum=0),
            warnings=tuple(warnings),
        )

    def require_api_key(self) -> None:
        """Valida a presença da chave apenas nos comandos que chamam a API."""
        if not self.openrouter_api_key:
            raise SettingsError(
                "OPENROUTER_API_KEY não definida.\n"
                "Copie `.env.example` para `.env` e preencha a chave "
                "(https://openrouter.ai/keys)."
            )
