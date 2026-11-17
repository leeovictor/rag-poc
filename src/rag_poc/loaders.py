"""Descoberta e leitura de arquivos (.pdf, .txt, .md, .mdx) em `Document`s."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from .errors import NoDocumentsError, PathNotFoundError, UnsupportedFileError

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".mdx", ".markdown"}
_PLAIN_TEXT_SUFFIXES = {".txt"}
_MARKDOWN_SUFFIXES = {".md", ".mdx", ".markdown"}

_FRONTMATTER_RE = re.compile(r"\A---\s*\n.*?\n---\s*\n?", re.DOTALL)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_MDX_EXPRESSION_COMMENT_RE = re.compile(r"\{/\*.*?\*/\}", re.DOTALL)
_MDX_ESM_RE = re.compile(r"^\s*(?:import|export)\s+")


@dataclass(frozen=True)
class FileError:
    path: Path
    message: str


@dataclass
class LoadReport:
    documents: list[Document] = field(default_factory=list)
    files: list[Path] = field(default_factory=list)
    errors: list[FileError] = field(default_factory=list)


def _is_hidden(path: Path, root: Path) -> bool:
    return any(part.startswith(".") for part in path.relative_to(root).parts)


def discover_files(path: Path) -> list[Path]:
    """Resolve um arquivo ou diretório (recursivo) para uma lista de arquivos suportados."""
    path = path.expanduser()
    if not path.exists():
        raise PathNotFoundError(path)

    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFileError(path, SUPPORTED_EXTENSIONS)
        return [path]

    files = [
        candidate
        for candidate in sorted(path.rglob("*"))
        if candidate.is_file()
        and candidate.suffix.lower() in SUPPORTED_EXTENSIONS
        and not _is_hidden(candidate, path)
    ]
    if not files:
        extensoes = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise NoDocumentsError(f"Nenhum arquivo ({extensoes}) encontrado em '{path}'.")
    return files


def _display_source(path: Path) -> str:
    """Caminho relativo ao diretório atual quando possível (mais legível)."""
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _read_text_file(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    # Fallback permissivo para arquivos em latin-1/windows-1252.
    return raw.decode("latin-1")


def _strip_mdx_esm(text: str) -> str:
    """Remove `import`/`export` de nível superior (ESM do MDX), preservando code fences."""
    lines: list[str] = []
    in_fence = False
    fence_marker = ""

    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            marker = stripped[:3]
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif marker == fence_marker:
                in_fence = False
            lines.append(line)
            continue
        if not in_fence and _MDX_ESM_RE.match(line):
            continue
        lines.append(line)

    return "\n".join(lines)


def _clean_markdown(text: str) -> str:
    """Remove ruído estrutural do Markdown/MDX mantendo prosa e exemplos de código."""
    text = text.lstrip("\ufeff")
    text = _FRONTMATTER_RE.sub("", text, count=1)
    text = _HTML_COMMENT_RE.sub("", text)
    text = _MDX_EXPRESSION_COMMENT_RE.sub("", text)
    text = _strip_mdx_esm(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _with_normalized_metadata(doc: Document, path: Path) -> Document:
    page = doc.metadata.get("page")
    metadata: dict[str, object] = {
        "source": _display_source(path),
        "file_name": path.name,
    }
    if isinstance(page, int):
        metadata["page"] = page
    return Document(page_content=doc.page_content, metadata=metadata)


def load_file(path: Path) -> list[Document]:
    """Lê um arquivo e retorna um `Document` por página (PDF) ou um único (texto)."""
    path = path.expanduser()
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        documents = PyPDFLoader(str(path)).load()
    elif suffix in _PLAIN_TEXT_SUFFIXES or suffix in _MARKDOWN_SUFFIXES:
        text = _read_text_file(path)
        if suffix in _MARKDOWN_SUFFIXES:
            text = _clean_markdown(text)
        documents = [Document(page_content=text, metadata={})]
    else:
        raise UnsupportedFileError(path, SUPPORTED_EXTENSIONS)

    documents = [
        _with_normalized_metadata(doc, path)
        for doc in documents
        if doc.page_content and doc.page_content.strip()
    ]
    if not documents:
        raise ValueError("nenhum texto extraível (PDF escaneado ou arquivo vazio?)")
    return documents


def load_documents(files: list[Path]) -> LoadReport:
    """Lê uma lista de arquivos, acumulando erros sem abortar o lote."""
    report = LoadReport()
    for file in files:
        try:
            report.documents.extend(load_file(file))
            report.files.append(file)
        except Exception as exc:  # noqa: BLE001 - erro por arquivo não deve abortar o lote
            report.errors.append(FileError(path=file, message=str(exc)))
    return report
