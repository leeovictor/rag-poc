"""Camada de apresentação no terminal (Rich): avisos, streaming, tabelas."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .config import Settings
from .indexer import IndexInfo, IndexStats
from .links import build_source_link
from .loaders import FileError
from .retriever import RetrievedChunk

console = Console()

_PAGE_DASH = "—"


def format_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, remainder = divmod(int(seconds), 60)
    return f"{minutes} min {remainder} s"


def show_warnings(warnings: Sequence[str]) -> None:
    for warning in warnings:
        console.print(f"[yellow]aviso:[/] {warning}")


def print_error(message: str, hint: str | None = None) -> None:
    body = message if hint is None else f"{message}\n\n[dim]{hint}[/]"
    console.print(Panel(body, title="[bold red]Erro[/]", border_style="red", expand=False))


def chat_intro(settings: Settings, hybrid_enabled: bool) -> None:
    mode = "híbrida (FAISS + BM25)" if hybrid_enabled else "vetorial (FAISS)"
    memory = (
        "desativada"
        if settings.chat_memory_turns <= 0
        else f"últimos {settings.chat_memory_turns} turno(s)"
    )
    links = (
        "desativados" if settings.citation_link_scheme == "none" else settings.citation_link_scheme
    )
    fanout = (
        f"até {settings.fanout_max_queries} consulta(s)"
        if settings.fanout_enabled
        else "desativado"
    )
    sufficiency = (
        "desligada"
        if settings.sufficiency_max_rounds <= 0
        else f"{settings.sufficiency_max_rounds} rodada(s)"
    )
    console.print(
        Panel(
            f"Modelo: [bold]{settings.llm_model}[/]\n"
            f"Busca: {mode} · k={settings.retrieval_k} · memória: {memory}\n"
            f"Fan-out: {fanout} · suficiência: {sufficiency} · links: {links}\n"
            "Pergunte, use [bold]limpar[/] para apagar o contexto, ou [bold]sair[/] / Ctrl+D.",
            title="[bold cyan]RAG Híbrido[/]",
            border_style="cyan",
            expand=False,
        )
    )


def ask_question() -> str | None:
    """Lê uma pergunta. Retorna `None` em EOF/Ctrl+C (encerrar)."""
    try:
        return console.input("\n[bold cyan]Você ›[/] ").strip()
    except (EOFError, KeyboardInterrupt):
        return None


_CITATION_LINK_RE = re.compile(r"\[(\d+)\](?!\()")
_FENCE_BLOCK_RE = re.compile(r"(```.*?```)", re.DOTALL)


def linkify_citations(
    text: str,
    chunks: Sequence[RetrievedChunk],
    link_scheme: str,
) -> str:
    """Transforma `[n]` em link Markdown para o arquivo da fonte, fora de code blocks."""
    if link_scheme == "none" or not chunks or "[" not in text:
        return text

    links: dict[str, str] = {}
    for position, chunk in enumerate(chunks, start=1):
        document = chunk.document
        source = str(document.metadata.get("source") or "")
        page = document.metadata.get("page")
        link = build_source_link(source, link_scheme, page if isinstance(page, int) else None)
        if link is not None:
            links[str(position)] = link

    if not links:
        return text

    def replace(match: re.Match[str]) -> str:
        number = match.group(1)
        url = links.get(number)
        if url is None:
            return match.group(0)
        return f"[[{number}]]({url})"

    parts = _FENCE_BLOCK_RE.split(text)
    for index in range(0, len(parts), 2):
        parts[index] = _CITATION_LINK_RE.sub(replace, parts[index])
    return "".join(parts)


def stream_answer(
    stream: Iterable[str],
    *,
    chunks: Sequence[RetrievedChunk] | None = None,
    link_scheme: str = "none",
) -> str:
    """Renderiza a resposta token a token em Markdown e devolve o texto final.

    O retorno é o texto bruto (sem links) para memória e extração de citações.
    """
    text = ""
    with Live(
        Markdown(""),
        console=console,
        refresh_per_second=10,
        vertical_overflow="visible",
    ) as live:
        for delta in stream:
            text += delta
            live.update(Markdown(linkify_citations(text, chunks or [], link_scheme)))
    return text


def _via_label(chunk: RetrievedChunk) -> str:
    return " + ".join(chunk.sources) if chunk.sources else _PAGE_DASH


def print_sources(
    chunks: Sequence[RetrievedChunk],
    cited: set[int],
    *,
    link_scheme: str = "none",
) -> None:
    if not chunks:
        console.print("[dim]Sem fontes recuperadas.[/]")
        return

    table = Table(title="Fontes", header_style="bold magenta", expand=False)
    table.add_column("#", justify="right", style="bold")
    table.add_column("Arquivo", overflow="fold")
    table.add_column("Página", justify="right")
    table.add_column("Via", justify="center")
    table.add_column("Citada", justify="center")
    show_queries = any(len(chunk.queries) > 1 for chunk in chunks)
    if show_queries:
        table.add_column("Subconsultas", justify="center")

    for position, chunk in enumerate(chunks, start=1):
        document = chunk.document
        source = str(document.metadata.get("source") or "")
        name = document.metadata.get("file_name") or source or "desconhecido"
        page = document.metadata.get("page")
        page_number = page if isinstance(page, int) else None
        page_label = str(page_number + 1) if page_number is not None else _PAGE_DASH

        name_text = Text(str(name))
        link = build_source_link(source, link_scheme, page_number)
        if link is not None:
            name_text.stylize(f"link {link}")
            name_text.stylize("underline")

        row: list[object] = [
            str(position),
            name_text,
            page_label,
            _via_label(chunk),
            "[green]sim[/]" if position in cited else _PAGE_DASH,
        ]
        if show_queries:
            row.append(str(len(chunk.queries)) if chunk.queries else _PAGE_DASH)
        table.add_row(*row)

    console.print(table)


def print_rewritten_query(query: str) -> None:
    console.print(f"[dim]Busca reformulada:[/] {escape(query)}")


def print_queries(queries: Sequence[str]) -> None:
    joined = " · ".join(escape(query) for query in queries)
    console.print(f"[dim]Consultas ({len(queries)}):[/] {joined}")


def print_planner_fallback() -> None:
    console.print(
        "[dim]Não foi possível planejar as consultas; usando a pergunta original na busca.[/]"
    )


def print_sufficiency_fallback() -> None:
    console.print("[dim]Verificação de suficiência indisponível; seguindo com o contexto atual.[/]")


def print_gap(missing: str, queries: Sequence[str]) -> None:
    detail = f" ({escape(missing)})" if missing else ""
    console.print(f"[yellow]Lacuna detectada{detail}:[/] +{len(queries)} consulta(s) de busca.")


def print_no_results(question: str, indexed_chunks: int) -> None:
    """Mensagem quando a recuperação não devolve nenhum trecho."""
    console.print(f'[yellow]Não encontrei trechos relevantes para[/] "{escape(question)}".')
    console.print(
        f"[dim]O índice tem {indexed_chunks} chunk(s). A resposta a seguir usará "
        "conhecimento geral e não terá citações; tente reformular se quiser uma "
        "resposta baseada na documentação.[/]"
    )


def print_chunks(chunks: Sequence[RetrievedChunk], *, limit: int = 400) -> None:
    for position, chunk in enumerate(chunks, start=1):
        content = chunk.document.page_content.strip().replace("\n", " ")
        preview = content if len(content) <= limit else content[:limit].rstrip() + "…"
        console.print(f"[bold cyan][{position}][/] [dim]{_via_label(chunk)}[/] {preview}")


def print_file_errors(errors: Sequence[FileError]) -> None:
    if not errors:
        return
    error_table = Table(title="Arquivos com erro", header_style="bold red", expand=False)
    error_table.add_column("Arquivo", overflow="fold")
    error_table.add_column("Motivo", overflow="fold")
    for error in errors:
        error_table.add_row(str(error.path), error.message)
    console.print(error_table)


def print_ingest_summary(
    *,
    stats: IndexStats,
    documents: int,
    files_read: int,
    files_total: int,
    elapsed: float,
    errors: Sequence[FileError],
) -> None:
    table = Table(title="Ingestão concluída", show_header=False, expand=False)
    table.add_column("Campo", style="bold")
    table.add_column("Valor")

    table.add_row("Modelo de embedding", stats.model)
    table.add_row("Dimensões", str(stats.dimensions))
    table.add_row("Arquivos lidos", f"{files_read}/{files_total}")
    table.add_row("Páginas/blocos de origem", str(documents))
    table.add_row("Chunks gerados", str(stats.chunks_total))
    table.add_row("Chunks novos", f"[green]{stats.chunks_added}[/]")
    table.add_row("Já indexados (ignorados)", str(stats.chunks_already_indexed))
    table.add_row("Duplicados no lote", str(stats.chunks_duplicate_in_batch))
    table.add_row("Vetores no índice", str(stats.total_vectors))
    table.add_row("Índice", str(stats.index_dir))
    table.add_row("Tempo", format_duration(elapsed))
    console.print(table)

    if errors:
        error_table = Table(title="Arquivos com erro", header_style="bold red", expand=False)
        error_table.add_column("Arquivo", overflow="fold")
        error_table.add_column("Motivo", overflow="fold")
        for error in errors:
            error_table.add_row(str(error.path), error.message)
        console.print(error_table)


def print_index_status(info: IndexInfo, settings: Settings) -> None:
    if info.manifest is None:
        console.print(
            f"[yellow]Nenhum índice em '{info.path}'.[/]\n"
            "Rode `rag ingest <arquivo-ou-diretório>` para criar."
        )
        return

    manifest = info.manifest
    table = Table(title="Índice FAISS", show_header=False, expand=False)
    table.add_column("Campo", style="bold")
    table.add_column("Valor")
    table.add_row("Diretório", str(info.path))
    table.add_row("Modelo de embedding", str(manifest.get("embedding_model", _PAGE_DASH)))
    table.add_row("Dimensões", str(manifest.get("embedding_dimensions", _PAGE_DASH)))
    table.add_row("Vetores", str(manifest.get("vectors", _PAGE_DASH)))
    table.add_row(
        "Chunk size / overlap", f"{manifest.get('chunk_size')} / {manifest.get('chunk_overlap')}"
    )
    table.add_row("Criado em", str(manifest.get("created_at", _PAGE_DASH)))
    table.add_row("Atualizado em", str(manifest.get("updated_at", _PAGE_DASH)))
    console.print(table)

    files = manifest.get("files")
    if isinstance(files, list) and files:
        file_table = Table(title="Arquivos indexados", expand=False)
        file_table.add_column("Arquivo", overflow="fold")
        for file in files:
            file_table.add_row(str(file))
        console.print(file_table)

    mode = "híbrida" if settings.hybrid_search else "vetorial"
    console.print(
        f"[dim]Configuração atual: {settings.embedding_model} · busca {mode} "
        f"· k={settings.retrieval_k}[/]"
    )


def _format_time_range(low: float, high: float) -> str:
    if low <= 0 and high <= 0:
        return _PAGE_DASH
    return f"{format_duration(low)} – {format_duration(high)}"


def print_ingest_estimate(
    *,
    files_total: int,
    files_read: int,
    documents: int,
    chunks: int,
    tokens: int,
    tokens_exact: bool,
    cost_usd: float,
    time_range: tuple[float, float],
    local_elapsed: float,
    chat_cost_usd: float,
    settings: Settings,
    errors: Sequence[FileError],
) -> None:
    """Resumo do `--dry-run`: nenhuma chamada de API é feita."""
    table = Table(title="Estimativa de ingestão (dry-run)", show_header=False, expand=False)
    table.add_column("Campo", style="bold")
    table.add_column("Valor")

    token_note = "" if tokens_exact else " [yellow](heurística chars/4)[/]"
    table.add_row("Modelo de embedding", settings.embedding_model)
    table.add_row("Arquivos lidos", f"{files_read}/{files_total}")
    table.add_row("Páginas/blocos de origem", str(documents))
    table.add_row("Chunks", str(chunks))
    table.add_row("Tokens de embedding", f"{tokens:,}{token_note}".replace(",", "."))
    table.add_row("Custo de embedding (est.)", f"US$ {cost_usd:.4f}")
    table.add_row("Tempo de embeddings (est.)", _format_time_range(*time_range))
    table.add_row("Processamento local (medido)", format_duration(local_elapsed))
    table.add_row(
        f"Custo por pergunta (k={settings.retrieval_k})",
        f"US$ {chat_cost_usd:.6f}",
    )
    table.add_row("Modelo de chat", settings.llm_model)
    console.print(table)
    console.print("[dim]Nenhuma chamada de API foi feita (--dry-run).[/]")
    print_file_errors(errors)
