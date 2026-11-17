"""Exceções de domínio da aplicação."""

from __future__ import annotations

from pathlib import Path


class RagError(Exception):
    """Erro base, com mensagem já pronta para ser exibida ao usuário."""


class SettingsError(RagError):
    """Configuração inválida ou ausente (variáveis de ambiente)."""


class PathNotFoundError(RagError):
    def __init__(self, path: Path) -> None:
        super().__init__(f"Caminho não encontrado: {path}")
        self.path = path


class UnsupportedFileError(RagError):
    def __init__(self, path: Path, supported: set[str]) -> None:
        extensoes = ", ".join(sorted(supported))
        super().__init__(f"Extensão não suportada em '{path}'. Formatos aceitos: {extensoes}.")
        self.path = path


class NoDocumentsError(RagError):
    """Nenhum documento com texto pôde ser extraído."""


class IndexNotFoundError(RagError):
    """Não existe índice FAISS persistido."""


class IndexModelMismatchError(RagError):
    """O índice foi criado com outro modelo/dimensão de embedding."""
