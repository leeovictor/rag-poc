from __future__ import annotations

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rank_bm25 import BM25Plus

from rag_poc.retriever import Retriever
from rag_poc.tokenizer import tokenize_en


def make_store(settings, fake_embeddings):
    documents = [
        Document(
            page_content="The rental contract expires in December.",
            metadata={"doc_id": "a", "source": "a.txt", "file_name": "a.txt"},
        ),
        Document(
            page_content="The cat sleeps by the window all afternoon.",
            metadata={"doc_id": "b", "source": "b.txt", "file_name": "b.txt"},
        ),
    ]
    store = FAISS.from_texts(
        [document.page_content for document in documents],
        fake_embeddings,
        metadatas=[document.metadata for document in documents],
        ids=["a", "b"],
    )
    return store, documents


def test_hybrid_prefers_lexical_match(settings, fake_embeddings) -> None:
    store, documents = make_store(settings, fake_embeddings)
    corpus = [tokenize_en(document.page_content) for document in documents]
    retriever = Retriever(
        store=store,
        settings=settings,
        documents=documents,
        bm25=BM25Plus(corpus),
    )

    results = retriever.retrieve("When does the rental contract expire?")

    assert retriever.hybrid_enabled is True
    assert results[0].document.metadata["doc_id"] == "a"
    assert "bm25" in results[0].sources
    assert "vetor" in results[0].sources


def test_vector_only_mode(settings, fake_embeddings) -> None:
    store, documents = make_store(settings, fake_embeddings)
    retriever = Retriever(
        store=store,
        settings=settings,
        documents=documents,
        bm25=None,
    )

    results = retriever.retrieve("cat")

    assert retriever.hybrid_enabled is False
    assert len(results) == settings.retrieval_k
    assert all(result.sources == ("vetor",) for result in results)


def test_retrieve_respects_k(settings, fake_embeddings) -> None:
    store, documents = make_store(settings, fake_embeddings)
    corpus = [tokenize_en(document.page_content) for document in documents]
    retriever = Retriever(
        store=store,
        settings=settings,
        documents=documents,
        bm25=BM25Plus(corpus),
    )

    results = retriever.retrieve("rental contract or cat")

    assert len(results) <= settings.retrieval_k


def test_retrieve_many_fuses_queries_and_tracks_provenance(settings, fake_embeddings) -> None:
    store, documents = make_store(settings, fake_embeddings)
    corpus = [tokenize_en(document.page_content) for document in documents]
    retriever = Retriever(
        store=store,
        settings=settings,
        documents=documents,
        bm25=BM25Plus(corpus),
    )

    results = retriever.retrieve_many(["rental contract", "cat window"])

    ids = {chunk.document.metadata["doc_id"] for chunk in results}
    assert ids == {"a", "b"}
    assert all(chunk.queries for chunk in results)
    by_id = {chunk.document.metadata["doc_id"]: chunk for chunk in results}
    assert "rental contract" in by_id["a"].queries
    assert "cat window" in by_id["b"].queries


def test_retrieve_many_dedupes_repeated_queries(settings, fake_embeddings) -> None:
    store, documents = make_store(settings, fake_embeddings)
    corpus = [tokenize_en(document.page_content) for document in documents]
    retriever = Retriever(
        store=store,
        settings=settings,
        documents=documents,
        bm25=BM25Plus(corpus),
    )

    results = retriever.retrieve_many(["cat", "CAT", "cat"])

    assert [chunk.document.metadata["doc_id"] for chunk in results] == [
        chunk.document.metadata["doc_id"] for chunk in retriever.retrieve("cat")
    ]
    assert all(chunk.queries == ("cat",) for chunk in results)


def test_retrieve_many_empty_queries(settings, fake_embeddings) -> None:
    store, documents = make_store(settings, fake_embeddings)
    retriever = Retriever(store=store, settings=settings, documents=documents, bm25=None)

    assert retriever.retrieve_many([]) == []
    assert retriever.retrieve_many(["   "]) == []
