from __future__ import annotations

import json
from dataclasses import replace

import pytest
from langchain_core.documents import Document

from rag_poc.errors import IndexModelMismatchError, IndexNotFoundError
from rag_poc.indexer import (
    chunk_documents,
    clear_index,
    doc_id_for,
    get_all_documents,
    index_chunks,
    load_vectorstore,
)


def make_documents() -> list[Document]:
    text = " ".join(f"palavra{index}" for index in range(120))
    return [
        Document(
            page_content=text,
            metadata={"source": "docs/a.txt", "file_name": "a.txt"},
        )
    ]


def test_chunk_documents_is_deterministic(settings) -> None:
    documents = make_documents()

    first = chunk_documents(settings, documents)
    second = chunk_documents(settings, documents)

    assert len(first) > 1
    assert [chunk.metadata["doc_id"] for chunk in first] == [
        chunk.metadata["doc_id"] for chunk in second
    ]
    assert all(isinstance(chunk.metadata["chunk_index"], int) for chunk in first)


def test_chunk_documents_preserves_page(settings) -> None:
    document = Document(
        page_content="texto " * 200,
        metadata={"source": "x.pdf", "file_name": "x.pdf", "page": 2},
    )

    chunks = chunk_documents(settings, [document])

    assert chunks
    assert all(chunk.metadata["page"] == 2 for chunk in chunks)


def test_doc_id_is_stable_and_sensitive_to_page() -> None:
    assert doc_id_for("a.txt", None, 0) == doc_id_for("a.txt", None, 0)
    assert doc_id_for("a.txt", None, 0) != doc_id_for("a.txt", 10, 0)
    assert doc_id_for("a.txt", None, 0) != doc_id_for("b.txt", None, 0)


def test_index_chunks_and_dedupe(settings, fake_embeddings) -> None:
    chunks = chunk_documents(settings, make_documents())

    first = index_chunks(settings, chunks, fake_embeddings)

    assert first.chunks_added == len(chunks)
    assert first.total_vectors == len(chunks)
    assert first.dimensions == 8
    assert (settings.faiss_index_dir / "index.faiss").exists()

    manifest = json.loads((settings.faiss_index_dir / "manifest.json").read_text())
    assert manifest["vectors"] == len(chunks)
    assert manifest["embedding_dimensions"] == 8
    assert manifest["embedding_model"] == settings.embedding_model

    second = index_chunks(settings, chunks, fake_embeddings)

    assert second.chunks_added == 0
    assert second.chunks_already_indexed == len(chunks)
    assert second.total_vectors == len(chunks)


def test_clear_index_is_idempotent(settings, fake_embeddings) -> None:
    chunks = chunk_documents(settings, make_documents())
    index_chunks(settings, chunks, fake_embeddings)

    assert clear_index(settings) is True
    assert not settings.faiss_index_dir.exists()
    assert clear_index(settings) is False


def test_index_roundtrip_and_document_store(settings, fake_embeddings) -> None:
    chunks = chunk_documents(settings, make_documents())
    index_chunks(settings, chunks, fake_embeddings)

    store = load_vectorstore(settings, fake_embeddings)
    documents = get_all_documents(store)

    assert len(documents) == len(chunks)
    assert {document.metadata["doc_id"] for document in documents} == {
        chunk.metadata["doc_id"] for chunk in chunks
    }


def test_load_vectorstore_requires_existing_index(settings, fake_embeddings) -> None:
    with pytest.raises(IndexNotFoundError):
        load_vectorstore(settings, fake_embeddings)


def test_embedding_model_mismatch_raises(settings, fake_embeddings) -> None:
    chunks = chunk_documents(settings, make_documents())
    index_chunks(settings, chunks, fake_embeddings)

    other_model = replace(settings, embedding_model="openai/text-embedding-3-large")
    with pytest.raises(IndexModelMismatchError):
        load_vectorstore(other_model, fake_embeddings)
