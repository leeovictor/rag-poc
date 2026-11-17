"""Fixtures compartilhadas."""

from __future__ import annotations

import hashlib

import pytest
from langchain_core.embeddings import Embeddings

from rag_poc.config import Settings


class FakeEmbeddings(Embeddings):
    """Embeddings determinísticos, sem chamadas de rede."""

    def __init__(self, size: int = 8) -> None:
        self.size = size

    def _embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [value / 255.0 for value in digest[: self.size]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        openrouter_api_key="sk-or-v1-test",
        openrouter_base_url="https://openrouter.ai/api/v1",
        llm_model="openai/gpt-4o-mini",
        embedding_model="openai/text-embedding-3-small",
        embedding_dimensions=None,
        embedding_batch_size=16,
        temperature=0.0,
        faiss_index_dir=tmp_path / "faiss_index",
        chunk_size=200,
        chunk_overlap=40,
        retrieval_k=2,
        hybrid_search=True,
        hybrid_weights=(0.5, 0.5),
        app_url="http://localhost",
        app_name="rag-poc",
        warnings=(),
    )


@pytest.fixture
def fake_embeddings() -> FakeEmbeddings:
    return FakeEmbeddings(size=8)
