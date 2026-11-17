"""Fábricas dos clientes OpenAI-compatíveis apontando para o OpenRouter."""

from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .config import Settings


def build_embeddings(settings: Settings) -> OpenAIEmbeddings:
    """Cliente de embeddings via OpenRouter (`/api/v1/embeddings`)."""
    kwargs: dict[str, Any] = {
        "model": settings.embedding_model,
        "api_key": settings.openrouter_api_key,
        "base_url": settings.openrouter_base_url,
        # Envia texto puro em vez de arrays de tokens: mais compatível com
        # endpoints OpenAI-compatíveis como o do OpenRouter.
        "check_embedding_ctx_length": False,
        "chunk_size": settings.embedding_batch_size,
        "max_retries": 3,
        "request_timeout": 60,
    }
    if settings.embedding_dimensions is not None:
        kwargs["dimensions"] = settings.embedding_dimensions
    return OpenAIEmbeddings(**kwargs)


def build_llm(settings: Settings) -> ChatOpenAI:
    """Cliente de chat via OpenRouter (/api/v1/chat/completions)."""
    headers: dict[str, str] = {}
    if settings.app_url:
        headers["HTTP-Referer"] = settings.app_url
    if settings.app_name:
        headers["X-Title"] = settings.app_name

    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        temperature=settings.temperature,
        request_timeout=120,
        max_retries=3,
        default_headers=headers or None,
    )
