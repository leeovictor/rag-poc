"""Construção de links clicáveis para abrir as fontes em um editor."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

VALID_SCHEMES = {"vscode", "file", "none"}


def build_source_link(source: str, scheme: str, page: int | None = None) -> str | None:
    """Retorna uma URI clicável para o arquivo ou `None` se não for possível.

    - ``vscode``: ``vscode://file/<caminho absoluto>`` (abre no VSCode);
    - ``file``: ``file:///<caminho>`` (app padrão; em PDFs tenta ``#page=N``);
    - ``none``: desativa os links.
    """
    if scheme == "none" or not source:
        return None

    path = Path(source).expanduser()
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError):
        return None

    if scheme == "file":
        uri = resolved.as_uri()
        if page is not None:
            uri += f"#page={page + 1}"
        return uri

    if scheme == "vscode":
        posix_path = resolved.as_posix()
        if not posix_path.startswith("/"):
            posix_path = "/" + posix_path
        return f"vscode://file{quote(posix_path, safe='/:')}"

    return None
