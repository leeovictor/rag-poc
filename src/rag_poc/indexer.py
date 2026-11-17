"""Chunking, deduplicação e persistência do índice FAISS."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings
from .errors import IndexModelMismatchError, IndexNotFoundError, RagError

INDEX_NAME = "index"
MANIFEST_NAME = "manifest.json"
MANIFEST_VERSION = 1


@dataclass(frozen=True)
class IndexStats:
    chunks_total: int
    chunks_added: int
    chunks_already_indexed: int
    chunks_duplicate_in_batch: int
    total_vectors: int
    dimensions: int
    model: str
    index_dir: Path


@dataclass(frozen=True)
class IndexInfo:
    path: Path
    manifest: dict[str, object] | None


def manifest_path(settings: Settings) -> Path:
    return settings.faiss_index_dir / MANIFEST_NAME


def index_files_exist(settings: Settings) -> bool:
    base = settings.faiss_index_dir
    return (base / f"{INDEX_NAME}.faiss").exists() and (base / f"{INDEX_NAME}.pkl").exists()


def doc_id_for(source: str, page: int | None, start_index: int) -> str:
    """ID determinístico do chunk: reingestões não duplicam o mesmo trecho."""
    page_key = "-" if page is None else str(page)
    raw = f"{source}|{page_key}|{start_index}".encode()
    return hashlib.sha256(raw).hexdigest()


def chunk_documents(settings: Settings, documents: list[Document]) -> list[Document]:
    """Divide documentos em chunks e enriquece o metadata com IDs determinísticos."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
    )
    raw_chunks = splitter.split_documents(documents)

    counters: dict[tuple[str, int | None], int] = defaultdict(int)
    seen: set[str] = set()
    chunks: list[Document] = []

    for chunk in raw_chunks:
        text = chunk.page_content.strip()
        if not text:
            continue

        source = str(chunk.metadata.get("source", "desconhecido"))
        page = chunk.metadata.get("page")
        page = page if isinstance(page, int) else None

        key = (source, page)
        chunk_index = counters[key]
        counters[key] += 1

        start_index = chunk.metadata.get("start_index")
        start_index = start_index if isinstance(start_index, int) else chunk_index

        doc_id = doc_id_for(source, page, start_index)
        if doc_id in seen:
            continue
        seen.add(doc_id)

        metadata: dict[str, object] = {
            "source": source,
            "file_name": chunk.metadata.get("file_name") or Path(source).name,
            "doc_id": doc_id,
            "chunk_index": chunk_index,
            "start_index": start_index,
        }
        if page is not None:
            metadata["page"] = page

        chunks.append(Document(page_content=text, metadata=metadata, id=doc_id))

    return chunks


def read_manifest(settings: Settings) -> dict[str, object] | None:
    path = manifest_path(settings)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RagError(
            f"Manifesto do índice ilegível em '{path}': {exc}. "
            "Rode `rag clear` e reingira os documentos."
        ) from exc
    return data if isinstance(data, dict) else None


def _validate_embedding_compat(
    settings: Settings,
    manifest: dict[str, object] | None,
    store: FAISS | None = None,
) -> None:
    if manifest is not None:
        stored_model = manifest.get("embedding_model")
        if isinstance(stored_model, str) and stored_model != settings.embedding_model:
            raise IndexModelMismatchError(
                f"O índice foi criado com o modelo '{stored_model}', mas a configuração atual "
                f"usa '{settings.embedding_model}'. Rode `rag clear` e reingira os documentos."
            )
        stored_dims = manifest.get("embedding_dimensions")
        if (
            settings.embedding_dimensions is not None
            and isinstance(stored_dims, int)
            and stored_dims != settings.embedding_dimensions
        ):
            raise IndexModelMismatchError(
                f"O índice usa {stored_dims} dimensões, mas EMBEDDING_DIMENSIONS="
                f"{settings.embedding_dimensions}. Rode `rag clear` e reingira os documentos."
            )

    if store is not None and settings.embedding_dimensions is not None:
        actual_dims = int(store.index.d)
        if actual_dims != settings.embedding_dimensions:
            raise IndexModelMismatchError(
                f"O índice FAISS usa {actual_dims} dimensões, mas EMBEDDING_DIMENSIONS="
                f"{settings.embedding_dimensions}. Rode `rag clear` e reingira os documentos."
            )


def open_vectorstore(settings: Settings, embeddings: Embeddings) -> FAISS | None:
    """Carrega o índice existente (validando o modelo) ou retorna `None`."""
    if not index_files_exist(settings):
        return None

    manifest = read_manifest(settings)
    _validate_embedding_compat(settings, manifest)
    try:
        store = FAISS.load_local(
            str(settings.faiss_index_dir),
            embeddings,
            index_name=INDEX_NAME,
            allow_dangerous_deserialization=True,
        )
    except Exception as exc:  # noqa: BLE001 - mensagem amigável para índice corrompido
        raise RagError(
            f"Falha ao carregar o índice FAISS em '{settings.faiss_index_dir}': {exc}. "
            "Rode `rag clear` e reingira os documentos."
        ) from exc
    _validate_embedding_compat(settings, manifest, store)
    return store


def load_vectorstore(settings: Settings, embeddings: Embeddings) -> FAISS:
    """Carrega o índice para consulta, exigindo que ele exista e tenha vetores."""
    if not index_files_exist(settings):
        raise IndexNotFoundError(
            f"Nenhum índice encontrado em '{settings.faiss_index_dir}'. "
            "Rode `rag ingest <arquivo-ou-diretório>` primeiro."
        )
    store = open_vectorstore(settings, embeddings)
    if store is None or store.index.ntotal == 0:
        raise IndexNotFoundError(
            f"O índice em '{settings.faiss_index_dir}' está vazio. "
            "Rode `rag ingest <arquivo-ou-diretório>` primeiro."
        )
    return store


def get_all_documents(store: FAISS) -> list[Document]:
    """Recupera todos os chunks do docstore (base do índice BM25)."""
    docstore = store.docstore
    raw = getattr(docstore, "_dict", None)
    if isinstance(raw, dict):
        return list(raw.values())
    return []


def _write_manifest(
    settings: Settings,
    manifest: dict[str, object] | None,
    store: FAISS,
    new_documents: list[Document],
) -> None:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    files = set(manifest.get("files", []) if manifest else [])
    files.update(str(doc.metadata.get("source", "")) for doc in new_documents)
    files.discard("")

    payload: dict[str, object] = {
        "version": MANIFEST_VERSION,
        "embedding_model": settings.embedding_model,
        "embedding_dimensions": int(store.index.d),
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "vectors": int(store.index.ntotal),
        "files": sorted(files),
        "created_at": manifest.get("created_at", now) if manifest else now,
        "updated_at": now,
    }
    settings.faiss_index_dir.mkdir(parents=True, exist_ok=True)
    manifest_path(settings).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def index_chunks(settings: Settings, chunks: list[Document], embeddings: Embeddings) -> IndexStats:
    """Adiciona chunks novos ao índice (ou cria o índice) e persiste em disco.

    Chunks já presentes (mesmo `doc_id`) são ignorados, tornando o `ingest`
    idempotente.
    """
    store = open_vectorstore(settings, embeddings)
    existing_ids: set[str] = (
        set(store.index_to_docstore_id.values()) if store is not None else set()
    )

    seen = set(existing_ids)
    new_documents: list[Document] = []
    new_ids: list[str] = []
    already_indexed = 0
    duplicate_in_batch = 0

    for chunk in chunks:
        doc_id = str(chunk.metadata["doc_id"])
        if doc_id in seen:
            if doc_id in existing_ids:
                already_indexed += 1
            else:
                duplicate_in_batch += 1
            continue
        seen.add(doc_id)
        new_documents.append(chunk)
        new_ids.append(doc_id)

    if new_documents:
        texts = [doc.page_content for doc in new_documents]
        metadatas = [doc.metadata for doc in new_documents]
        if store is None:
            store = FAISS.from_texts(texts, embeddings, metadatas=metadatas, ids=new_ids)
        else:
            store.add_texts(texts, metadatas=metadatas, ids=new_ids)
        store.save_local(str(settings.faiss_index_dir), index_name=INDEX_NAME)
        _write_manifest(settings, read_manifest(settings), store, new_documents)

    total_vectors = int(store.index.ntotal) if store is not None else 0
    dimensions = int(store.index.d) if store is not None else 0

    return IndexStats(
        chunks_total=len(chunks),
        chunks_added=len(new_documents),
        chunks_already_indexed=already_indexed,
        chunks_duplicate_in_batch=duplicate_in_batch,
        total_vectors=total_vectors,
        dimensions=dimensions,
        model=settings.embedding_model,
        index_dir=settings.faiss_index_dir,
    )


def clear_index(settings: Settings) -> bool:
    """Remove o diretório do índice. Retorna `False` se ele nem existia."""
    path = settings.faiss_index_dir
    if not path.exists():
        return False
    shutil.rmtree(path)
    return True


def get_index_info(settings: Settings) -> IndexInfo:
    manifest = read_manifest(settings) if manifest_path(settings).exists() else None
    return IndexInfo(path=settings.faiss_index_dir, manifest=manifest)
