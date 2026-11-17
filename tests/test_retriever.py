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
