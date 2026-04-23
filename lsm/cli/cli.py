"""Top-level CLI."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from lsm import __version__
from lsm.config import get_settings
from lsm.core.embeddings import get_default_embedder
from lsm.core.memory import build_store
from lsm.core.state import StateStore
from lsm.logging_setup import setup_logging
from lsm.runtime.llm_backend import ChatMessage, GenerateOptions, build_backend

app = typer.Typer(
    name="lsm",
    help="Living State Machine: a Stentor-inspired cognitive wrapper around local LLMs.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
console = Console()
log = logging.getLogger(__name__)


@app.callback()
def _root(
    log_level: str = typer.Option(
        "INFO", "--log-level", "-l", help="Logging level: DEBUG/INFO/WARNING/ERROR."
    ),
) -> None:
    setup_logging(log_level)


@app.command()
def version() -> None:
    """Print the installed lsm version and exit."""
    console.print(f"lsm {__version__}")


@app.command()
def serve(
    host: str | None = typer.Option(None, help="Bind host. Default from LSM_HOST."),
    port: int | None = typer.Option(None, help="Bind port. Default from LSM_PORT."),
    model: str | None = typer.Option(None, help="Model tag override (e.g. qwen2.5:7b-instruct-q8_0)."),
    reload: bool = typer.Option(False, help="Autoreload (dev only)."),
    stop: bool = typer.Option(False, "--stop", help="Unload the current model from Ollama and exit."),
) -> None:
    """Start the OpenAI-compatible API server.

    Use ``--stop`` to unload the model without running the server (frees VRAM).
    """
    import uvicorn

    settings = get_settings()
    if model:
        settings.model = model

    if stop:
        asyncio.run(_stop_model(settings.backend, settings.model, settings.ollama_host))
        console.print(f"[yellow]requested unload of {settings.model}[/yellow]")
        return

    from lsm.serve.api import create_app

    app_ = create_app(settings)
    uvicorn.run(
        app_,
        host=host or settings.host,
        port=port or settings.port,
        log_level=settings.log_level.lower(),
        reload=reload,
    )


async def _stop_model(backend_name: str, model: str, ollama_host: str) -> None:
    backend = build_backend(backend_name, model=model, ollama_host=ollama_host)
    try:
        if hasattr(backend, "unload"):
            await backend.unload()  # type: ignore[attr-defined]
    finally:
        await backend.close()


@app.command()
def chat(
    model: str | None = typer.Option(None, help="Model tag override."),
    system: str | None = typer.Option(
        None, help="Optional system prompt override for this session."
    ),
    session_id: str = typer.Option("cli", help="Session id recorded in the episodic log."),
) -> None:
    """Interactive REPL. Talks directly to the backend (no server round-trip).

    Use ``/exit`` (or Ctrl+C) to quit.
    """
    asyncio.run(_chat_loop(model=model, system=system, session_id=session_id))


async def _chat_loop(model: str | None, system: str | None, session_id: str) -> None:
    from lsm.core.episodic import EpisodicLog
    from lsm.core.pipeline import Pipeline, PipelineConfig

    settings = get_settings()
    effective_model = model or settings.model
    backend = build_backend(
        settings.backend, model=effective_model, ollama_host=settings.ollama_host
    )
    episodic = EpisodicLog(settings.memory_dir / "episodes.sqlite")
    pipe = Pipeline(
        backend=backend,
        episodic=episodic,
        config=PipelineConfig(
            session_id=session_id,
            backend_name=settings.backend,
            model=effective_model,
        ),
    )

    history: list[ChatMessage] = []
    if system:
        history.append(ChatMessage(role="system", content=system))

    console.print(f"[bold cyan]lsm chat[/bold cyan]  model={effective_model}  session={session_id}")
    console.print("Type [dim]/exit[/dim] to quit.\n")

    try:
        while True:
            try:
                user = Prompt.ask("[green]you[/green]")
            except (EOFError, KeyboardInterrupt):
                break
            if user.strip() in {"/exit", "/quit"}:
                break
            history.append(ChatMessage(role="user", content=user))
            console.print("[magenta]ai[/magenta] ", end="")
            buf: list[str] = []
            async for piece in pipe.stream(history, GenerateOptions()):
                console.print(piece, end="")
                buf.append(piece)
            console.print()
            history.append(ChatMessage(role="assistant", content="".join(buf)))
    finally:
        await backend.close()
        episodic.close()


state_app = typer.Typer(name="state", help="Inspect and manage the hidden state s_t.")
snapshot_app = typer.Typer(
    name="snapshot", help="Create and restore named snapshots of the state directory."
)
app.add_typer(state_app, name="state")
app.add_typer(snapshot_app, name="snapshot")


@state_app.command("show")
def state_show() -> None:
    """Print the current state summary and metadata."""
    settings = get_settings()
    store = StateStore(settings.home)
    s = store.load()
    t = Table(show_header=False, box=None)
    t.add_row("revision", str(s.revision))
    t.add_row("updated_at", f"{s.updated_at:.3f}")
    t.add_row("vector_dim", str(int(s.vector.size)))
    t.add_row("text_chars", str(len(s.text)))
    console.print(t)
    if s.text:
        console.print(Panel(s.text, title="s_t.text", border_style="cyan"))
    else:
        console.print("[dim](state is empty)[/dim]")


@state_app.command("clear")
def state_clear(
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation."),
) -> None:
    """Reset the hidden state to empty. Snapshot first if you value it."""
    if not yes:
        confirmed = Prompt.ask("Clear current state? type 'yes' to confirm", default="no")
        if confirmed.strip().lower() != "yes":
            console.print("aborted")
            raise typer.Exit(code=1)
    settings = get_settings()
    store = StateStore(settings.home)
    from lsm.core.state import State

    store.save(State())
    console.print("[yellow]state cleared[/yellow]")


@snapshot_app.command("create")
def snapshot_create(tag: str = typer.Argument(..., help="Snapshot tag.")) -> None:
    settings = get_settings()
    store = StateStore(settings.home)
    path = store.snapshot(tag)
    console.print(f"[green]snapshot created:[/green] {path}")


@snapshot_app.command("restore")
def snapshot_restore(tag: str = typer.Argument(..., help="Snapshot tag.")) -> None:
    settings = get_settings()
    store = StateStore(settings.home)
    store.restore(tag)
    console.print(f"[green]restored snapshot:[/green] {tag}")


@snapshot_app.command("list")
def snapshot_list() -> None:
    settings = get_settings()
    store = StateStore(settings.home)
    tags = store.list_snapshots()
    if not tags:
        console.print("[dim](no snapshots)[/dim]")
        return
    for t in tags:
        console.print(f"- {t}")


memory_app = typer.Typer(name="memory", help="Inspect and manage durable memory.")
app.add_typer(memory_app, name="memory")


def _open_memory_store():
    settings = get_settings()
    embedder = get_default_embedder(settings.embed_model)
    return build_store(
        "lancedb", root=settings.memory_dir / "durable.lance", embedder=embedder
    )


@memory_app.command("count")
def memory_count() -> None:
    store = _open_memory_store()
    try:
        console.print(store.count())
    finally:
        store.close()


@memory_app.command("query")
def memory_query(
    text: str = typer.Argument(..., help="Query text."),
    top_k: int = typer.Option(5, "--top-k", "-k"),
) -> None:
    store = _open_memory_store()
    try:
        hits = store.query(text, top_k=top_k)
    finally:
        store.close()
    if not hits:
        console.print("[dim](no hits)[/dim]")
        return
    t = Table(title=f"top {len(hits)} for '{text}'")
    t.add_column("score", justify="right")
    t.add_column("source")
    t.add_column("text", overflow="fold")
    for h in hits:
        t.add_row(f"{h.score:.3f}", h.entry.source, h.entry.text)
    console.print(t)


@memory_app.command("delete")
def memory_delete(
    source: str = typer.Option(None, "--source", help="Delete all entries from this source."),
    id_: str = typer.Option(None, "--id", help="Delete the entry with this id."),
) -> None:
    if not source and not id_:
        console.print("Provide --source or --id.")
        raise typer.Exit(code=2)
    store = _open_memory_store()
    try:
        if source:
            n = store.delete_by_source(source)
            console.print(f"[yellow]deleted by source {source}[/yellow] (rows={n})")
        if id_:
            n = store.delete([id_])
            console.print(f"[yellow]deleted id {id_}[/yellow] (rows={n})")
    finally:
        store.close()


@app.command()
def ingest(
    paths: list[Path] = typer.Argument(..., help="Files to ingest."),
    source: str = typer.Option(
        None, "--source", help="Source tag. Defaults to each file's path."
    ),
    chunk_chars: int = typer.Option(1200, "--chunk-chars", help="Chunk size in characters."),
    overlap: int = typer.Option(200, "--overlap", help="Overlap between chunks in characters."),
) -> None:
    """Ingest files into durable memory. Splits by character-count chunks."""
    store = _open_memory_store()
    total = 0
    try:
        for p in paths:
            if not p.exists():
                console.print(f"[red]missing:[/red] {p}")
                continue
            text = p.read_text(encoding="utf-8", errors="replace")
            src = source or str(p.resolve())
            chunks = _chunk(text, chunk_chars, overlap)
            for i, ch in enumerate(chunks):
                store.put(
                    ch,
                    source=src,
                    tags=["ingest"],
                    meta={"path": str(p), "chunk": i, "chunk_total": len(chunks)},
                )
            total += len(chunks)
            console.print(f"[green]{p}[/green] -> {len(chunks)} chunks")
    finally:
        store.close()
    console.print(f"ingested {total} chunks total")


def _chunk(text: str, size: int, overlap: int) -> list[str]:
    if size <= 0:
        return [text]
    step = max(size - overlap, 1)
    out: list[str] = []
    i = 0
    while i < len(text):
        out.append(text[i : i + size])
        i += step
    return out


eval_app = typer.Typer(name="eval", help="Run benchmarks and compare systems.")
app.add_typer(eval_app, name="eval")


@eval_app.command("run")
def eval_run(
    config: Path = typer.Option(..., "--config", "-c", help="Path to experiment YAML."),
    out: Path = typer.Option(
        Path("experiments"), "--out", "-o", help="Output root directory."
    ),
    run_id: str | None = typer.Option(None, "--run-id"),
) -> None:
    """Run an experiment config and write results under ``experiments/<run_id>/``."""
    from lsm.eval.runner import run_from_config_sync

    run_dir = run_from_config_sync(config_path=config, out_root=out, run_id=run_id)
    console.print(f"[green]results:[/green] {run_dir}")


if __name__ == "__main__":
    app()
