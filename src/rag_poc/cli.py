"""Interface de linha de comando (Typer + Rich)."""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, TypeVar

import openai
import typer
from langchain_core.documents import Document
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
)

from . import estimation, indexer, loaders, ui
from .chain import (
    EMPTY_CONTEXT_PLACEHOLDER,
    build_answer_chain,
    build_condense_chain,
    condense_question,
    extract_citations,
    format_context,
)
from .clients import build_embeddings
from .config import Settings
from .errors import NoDocumentsError, RagError
from .memory import ConversationMemory
from .retriever import Retriever

F = TypeVar("F", bound=Callable[..., Any])

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
    help="PoC de RAG híbrido (FAISS + BM25) com embeddings e LLM via OpenRouter.",
)

_API_MESSAGES: dict[int, tuple[str, str]] = {
    401: (
        "Chave do OpenRouter inválida ou não autorizada (HTTP 401).",
        "Confira OPENROUTER_API_KEY no arquivo .env.",
    ),
    402: (
        "Créditos insuficientes no OpenRouter (HTTP 402).",
        "Adicione créditos em https://openrouter.ai/settings/credits.",
    ),
    403: (
        "Acesso negado pelo OpenRouter (HTTP 403).",
        "Verifique permissões e limites da sua conta.",
    ),
    404: (
        "Recurso não encontrado no OpenRouter (HTTP 404).",
        "Confirme o id do modelo com prefixo de provider, ex.: openai/gpt-4o-mini.",
    ),
    429: (
        "Limite de requisições atingido no OpenRouter (HTTP 429).",
        "Aguarde alguns segundos e tente novamente.",
    ),
}


def _api_error_message(status_code: int) -> tuple[str, str]:
    return _API_MESSAGES.get(
        status_code,
        (
            f"Erro na API do OpenRouter (HTTP {status_code}).",
            "Tente novamente; se persistir, rode com --debug para ver o detalhe.",
        ),
    )


def _maybe_debug(debug: bool) -> None:
    if debug:
        ui.console.print_exception(show_locals=False)


def handle_errors(func: F) -> F:
    """Converte exceções conhecidas em mensagens amigáveis e códigos de saída."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        debug = bool(kwargs.get("debug", False))
        try:
            return func(*args, **kwargs)
        except (typer.Exit, typer.Abort):
            raise
        except RagError as exc:
            ui.print_error(str(exc))
            _maybe_debug(debug)
            raise typer.Exit(code=1) from None
        except openai.APIStatusError as exc:
            message, hint = _api_error_message(exc.status_code)
            ui.print_error(f"{message}\n[dim]{exc}[/]", hint)
            _maybe_debug(debug)
            raise typer.Exit(code=1) from None
        except openai.APIConnectionError as exc:
            ui.print_error(
                "Não foi possível conectar ao OpenRouter.",
                f"{exc}\nVerifique sua conexão e tente novamente.",
            )
            _maybe_debug(debug)
            raise typer.Exit(code=1) from None
        except openai.OpenAIError as exc:
            ui.print_error(f"Erro na API do OpenRouter: {exc}")
            _maybe_debug(debug)
            raise typer.Exit(code=1) from None
        except KeyboardInterrupt:
            ui.console.print("\n[dim]Interrompido pelo usuário.[/]")
            raise typer.Exit(code=130) from None
        except Exception as exc:  # noqa: BLE001 - fallback amigável da CLI
            ui.print_error(
                f"Erro inesperado: {exc}",
                "Rode novamente com --debug para ver o traceback completo.",
            )
            _maybe_debug(debug)
            raise typer.Exit(code=1) from None

    return wrapper  # type: ignore[return-value]


def _load_and_chunk(
    settings: Settings,
    path: Path,
) -> tuple[list[Path], list[Document], list[Document], list[loaders.FileError]]:
    """Descobre, lê e faz chunk dos arquivos (sem chamar a API)."""
    files = loaders.discover_files(path)
    ui.console.print(f"[dim]{len(files)} arquivo(s) encontrado(s) em '{path}'.[/]")

    documents: list[Document] = []
    errors: list[loaders.FileError] = []
    progress_columns = (
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
    )
    with Progress(*progress_columns, console=ui.console, transient=True) as progress:
        task = progress.add_task("Lendo arquivos", total=len(files))
        for file in files:
            progress.update(task, description=f"Lendo [bold]{file.name}[/]")
            try:
                documents.extend(loaders.load_file(file))
            except Exception as exc:  # noqa: BLE001 - erro por arquivo segue o lote
                errors.append(loaders.FileError(path=file, message=str(exc)))
            progress.advance(task)

    if not documents:
        raise NoDocumentsError(
            "Nenhum texto pôde ser extraído dos arquivos informados. "
            "PDFs escaneados precisam de OCR antes da ingestão."
        )

    chunks = indexer.chunk_documents(settings, documents)
    return files, documents, chunks, errors


@app.command()
@handle_errors
def ingest(
    path: Path = typer.Argument(
        ...,
        metavar="CAMINHO",
        help="Arquivo .pdf/.txt/.md/.mdx ou diretório com arquivos.",
    ),
    rebuild: bool = typer.Option(
        False, "--rebuild", help="Apaga o índice existente antes de indexar."
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Estima chunks, tokens, custo e tempo sem chamar a API.",
    ),
    debug: bool = typer.Option(
        False, "--debug", help="Exibe o traceback completo em caso de erro."
    ),
) -> None:
    """Indexa arquivos (PDF/TXT/Markdown) em um índice FAISS local (idempotente)."""
    started = time.perf_counter()
    settings = Settings.from_env()
    if not dry_run:
        settings.require_api_key()
    ui.show_warnings(settings.warnings)

    if rebuild:
        if dry_run:
            ui.console.print("[dim]--rebuild ignorado em --dry-run (nada é alterado).[/]")
        elif indexer.clear_index(settings):
            ui.console.print(
                f"[yellow]Índice anterior removido de '{settings.faiss_index_dir}'.[/]"
            )

    files, documents, chunks, errors = _load_and_chunk(settings, path)
    local_elapsed = time.perf_counter() - started

    if dry_run:
        texts = [chunk.page_content for chunk in chunks]
        tokens, exact = estimation.count_tokens(texts)
        cost_usd = estimation.estimate_embedding_cost_usd(settings.embedding_model, tokens)
        time_range = estimation.estimate_embedding_time_seconds(
            len(chunks), settings.embedding_batch_size
        )
        average_chunk_tokens = tokens / len(chunks) if chunks else 0.0
        chat_input = int(
            average_chunk_tokens * settings.retrieval_k + estimation.PROMPT_OVERHEAD_TOKENS
        )
        chat_cost = estimation.estimate_chat_cost_usd(settings.llm_model, chat_input)
        ui.print_ingest_estimate(
            files_total=len(files),
            files_read=len(files) - len(errors),
            documents=len(documents),
            chunks=len(chunks),
            tokens=tokens,
            tokens_exact=exact,
            cost_usd=cost_usd,
            time_range=time_range,
            local_elapsed=local_elapsed,
            chat_cost_usd=chat_cost,
            settings=settings,
            errors=errors,
        )
        return

    embeddings = build_embeddings(settings)
    with ui.console.status(
        "[bold cyan]Gerando embeddings e gravando o índice FAISS...[/]", spinner="dots"
    ):
        stats = indexer.index_chunks(settings, chunks, embeddings)

    ui.print_ingest_summary(
        stats=stats,
        documents=len(documents),
        files_read=len(files) - len(errors),
        files_total=len(files),
        elapsed=time.perf_counter() - started,
        errors=errors,
    )


@app.command()
@handle_errors
def chat(
    k: int | None = typer.Option(
        None, "--k", min=1, help="Número de trechos recuperados por pergunta."
    ),
    show_chunks: bool = typer.Option(
        False, "--show-chunks", help="Exibe um preview dos trechos recuperados."
    ),
    debug: bool = typer.Option(
        False, "--debug", help="Exibe o traceback completo em caso de erro."
    ),
) -> None:
    """Abre um chat interativo sobre os documentos indexados."""
    settings = Settings.from_env()
    settings.require_api_key()
    ui.show_warnings(settings.warnings)

    if k is not None:
        settings = replace(settings, retrieval_k=k)

    with ui.console.status(
        "[bold cyan]Carregando índice e preparando a busca híbrida...[/]", spinner="dots"
    ):
        retriever = Retriever.build(settings)
        chain = build_answer_chain(settings)
        condense_chain = build_condense_chain(settings)

    memory = ConversationMemory(settings.chat_memory_turns)
    ui.chat_intro(settings, hybrid_enabled=retriever.hybrid_enabled)

    while True:
        question = ui.ask_question()
        if question is None:
            ui.console.print("\n[dim]Encerrando chat.[/]")
            break
        if not question:
            continue
        command = question.lower()
        if command in {"sair", "exit", "quit"}:
            ui.console.print("[dim]Encerrando chat.[/]")
            break
        if command in {"limpar", "reset"}:
            memory.clear()
            ui.console.print("[dim]Contexto da conversa apagado.[/]")
            continue

        started = time.perf_counter()
        try:
            condense_failed = False
            with ui.console.status("[bold cyan]Buscando trechos relevantes...[/]", spinner="dots"):
                search_query = question
                if memory.turn_count > 0:
                    try:
                        search_query = condense_question(condense_chain, question, memory.as_text())
                    except openai.OpenAIError:
                        condense_failed = True
                        search_query = question
                chunks = retriever.retrieve(search_query)

            if condense_failed:
                ui.print_condense_fallback()
            if search_query.strip() != question.strip():
                ui.print_rewritten_query(search_query)

            if not chunks:
                ui.print_no_results(question, retriever.document_count)
            if show_chunks:
                ui.print_chunks(chunks)

            ui.console.print(
                f"\n[bold cyan]Resposta[/] [dim]({len(chunks)} trecho(s) recuperado(s))[/]"
            )
            answer = ui.stream_answer(
                chain.stream(
                    {
                        "history": memory.messages(),
                        "context": format_context(chunks) or EMPTY_CONTEXT_PLACEHOLDER,
                        "question": question,
                    }
                )
            )
            memory.add_turn(question, answer)
            cited = extract_citations(answer, len(chunks))
            ui.print_sources(chunks, cited, link_scheme=settings.citation_link_scheme)
            ui.console.print(f"[dim]{ui.format_duration(time.perf_counter() - started)}[/]")
        except openai.APIStatusError as exc:
            message, hint = _api_error_message(exc.status_code)
            ui.print_error(f"{message}\n[dim]{exc}[/]", hint)
            _maybe_debug(debug)
        except openai.APIConnectionError as exc:
            ui.print_error("Falha de conexão com o OpenRouter.", str(exc))
            _maybe_debug(debug)
        except openai.OpenAIError as exc:
            ui.print_error(f"Erro na API do OpenRouter: {exc}")
            _maybe_debug(debug)


@app.command()
@handle_errors
def clear(
    yes: bool = typer.Option(False, "--yes", "-y", help="Não pedir confirmação."),
    debug: bool = typer.Option(
        False, "--debug", help="Exibe o traceback completo em caso de erro."
    ),
) -> None:
    """Remove o índice local para reiniciar os testes do zero."""
    settings = Settings.from_env()
    path = settings.faiss_index_dir

    if not path.exists():
        ui.console.print(f"[dim]Nada para remover: '{path}' não existe.[/]")
        return

    if not yes:
        typer.confirm(f"Remover o índice em '{path}'?", abort=True)

    indexer.clear_index(settings)
    ui.console.print(f"[green]Índice removido de '{path}'.[/]")


@app.command()
@handle_errors
def status(
    debug: bool = typer.Option(
        False, "--debug", help="Exibe o traceback completo em caso de erro."
    ),
) -> None:
    """Mostra o manifesto e a configuração atual do índice."""
    settings = Settings.from_env()
    ui.show_warnings(settings.warnings)
    ui.print_index_status(indexer.get_index_info(settings), settings)


if __name__ == "__main__":
    app()
