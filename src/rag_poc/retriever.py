"""Recuperação híbrida: FAISS (vetorial) + BM25 (lexical), fundidos por RRF."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rank_bm25 import BM25Plus

from .clients import build_embeddings
from .config import Settings
from .indexer import get_all_documents, load_vectorstore
from .tokenizer import tokenize_en

# Constante clássica do Reciprocal Rank Fusion.
RRF_K = 60
_SOURCE_ORDER = ("vetor", "bm25")


@dataclass(frozen=True)
class RetrievedChunk:
    document: Document
    score: float
    sources: tuple[str, ...]


def _doc_key(document: Document) -> str:
    doc_id = document.metadata.get("doc_id")
    if isinstance(doc_id, str) and doc_id:
        return doc_id
    raw = f"{document.metadata.get('source', '')}|{document.page_content}".encode()
    return hashlib.sha256(raw).hexdigest()


class Retriever:
    """Busca híbrida com fusão por Reciprocal Rank Fusion (RRF).

    Quando ``HYBRID_SEARCH=false`` ou não há documentos suficientes, opera
    apenas com similaridade vetorial.
    """

    def __init__(
        self,
        store: FAISS,
        settings: Settings,
        documents: list[Document],
        bm25: BM25Plus | None = None,
    ) -> None:
        self._store = store
        self._settings = settings
        self._documents = documents
        self._bm25 = bm25

    @classmethod
    def build(cls, settings: Settings) -> Retriever:
        embeddings = build_embeddings(settings)
        store = load_vectorstore(settings, embeddings)
        documents = get_all_documents(store)

        bm25: BM25Plus | None = None
        if settings.hybrid_search and len(documents) >= 2:
            corpus = [tokenize_en(document.page_content) for document in documents]
            if any(corpus):
                bm25 = BM25Plus(corpus)

        return cls(store=store, settings=settings, documents=documents, bm25=bm25)

    @property
    def hybrid_enabled(self) -> bool:
        return self._bm25 is not None

    @property
    def document_count(self) -> int:
        return len(self._documents)

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        k = self._settings.retrieval_k
        weights = self._settings.hybrid_weights if self._bm25 is not None else (1.0, 0.0)

        fused: dict[str, dict[str, object]] = {}

        def register(document: Document, weight: float, source: str) -> None:
            key = _doc_key(document)
            entry = fused.setdefault(key, {"document": document, "score": 0.0, "sources": set()})
            entry["score"] = float(entry["score"]) + weight
            sources = entry["sources"]
            assert isinstance(sources, set)
            sources.add(source)

        vector_hits = self._store.similarity_search_with_score(query, k=k)
        for rank, (document, _distance) in enumerate(vector_hits):
            register(document, weights[0] / (RRF_K + rank + 1), "vetor")

        if self._bm25 is not None:
            tokens = tokenize_en(query)
            if tokens:
                scores = self._bm25.get_scores(tokens)
                ranked = sorted(
                    range(len(self._documents)),
                    key=lambda index: float(scores[index]),
                    reverse=True,
                )
                for rank, index in enumerate(ranked):
                    if rank >= k or scores[index] <= 0:
                        break
                    register(
                        self._documents[index],
                        weights[1] / (RRF_K + rank + 1),
                        "bm25",
                    )

        ordered = sorted(fused.values(), key=lambda entry: float(entry["score"]), reverse=True)[:k]

        return [
            RetrievedChunk(
                document=entry["document"],  # type: ignore[arg-type]
                score=float(entry["score"]),
                sources=tuple(
                    source
                    for source in _SOURCE_ORDER
                    if source in entry["sources"]  # type: ignore[operator]
                ),
            )
            for entry in ordered
        ]
