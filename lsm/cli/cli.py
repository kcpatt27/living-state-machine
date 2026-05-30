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
from lsm.core.fact_categories import FactCategoryRegistry
from lsm.core.facts import CAT_ASSISTANT_NOTES, Fact, FactStore
from lsm.core.identity import IdentityStore
from lsm.core.memory import MemoryEntry, build_store
from lsm.core.normalize import EmbeddingTermNormalizer
from lsm.core.state import StateStore
from lsm.core.verifier import RuleBasedVerifier, VerifierContext
from lsm.core.verifier_features import feature_dict
from lsm.core.verifier_log import VerifierLog
from lsm.logging_setup import setup_logging
from lsm.runtime.llm_backend import ChatMessage, GenerateOptions, build_backend
from lsm.verifier_data.context_text import render_context_text
from lsm.verifier_data.export import context_from_snapshot, export_verifier_dataset
from lsm.verifier_data.schema import VerifierDatasetRow, load_rows, write_rows

app = typer.Typer(
    name="lsm",
    help="Living State Machine: a Stentor-inspired cognitive wrapper around local LLMs.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
verify_app = typer.Typer(
    name="verify",
    help="Verifier tools: score, inspect history, manual labels, dataset export.",
    invoke_without_command=True,
    no_args_is_help=False,
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


app.add_typer(verify_app, name="verify")


@verify_app.callback(invoke_without_command=True)
def verify(
    ctx: typer.Context,
    explain: bool = typer.Option(
        False, "--explain", help="Show per-violation evidence in --history output."
    ),
    history: int | None = typer.Option(
        None,
        "--history",
        "-n",
        help="Print the last N verifier telemetry events from verifier_log.sqlite.",
    ),
) -> None:
    """Show history directly or run a verify subcommand."""
    if ctx.invoked_subcommand is not None:
        return
    settings = get_settings()
    if history is not None:
        _verify_history(settings.home, limit=history, explain=explain)
        return
    console.print("Use `lsm verify score <text>` or `lsm verify --history N`.")
    raise typer.Exit(code=2)


@verify_app.command("score")
def verify_score(
    text: str = typer.Argument(..., help="Candidate text to score."),
    source: str = typer.Option(
        "manual",
        "--source",
        help="Label for this candidate (summarizer | writer | reply | manual).",
    ),
    explain: bool = typer.Option(
        False, "--explain", help="Show per-violation evidence (fact/memory ids)."
    ),
    threshold_reprompt: float = typer.Option(1.0, "--reprompt-threshold"),
    threshold_reject: float = typer.Option(2.5, "--reject-threshold"),
) -> None:
    """Score candidate text against the live verifier context."""
    _score_candidate(
        text=text,
        source=source,
        explain=explain,
        threshold_reprompt=threshold_reprompt,
        threshold_reject=threshold_reject,
    )


@verify_app.command("label")
def verify_label(
    limit: int = typer.Option(50, "--limit", "-n", min=1, help="Max events to review."),
    out: Path = typer.Option(
        Path("data/verifier_gold_v1.jsonl"),
        "--out",
        help="Output gold-label JSONL path.",
    ),
    append: bool = typer.Option(True, "--append/--overwrite", help="Append to existing file."),
) -> None:
    """Manually label verifier events (accept/reprompt/reject) into a gold set."""
    settings = get_settings()
    log_path = settings.home / "verifier_log.sqlite"
    if not log_path.exists():
        console.print(f"[dim]no verifier log at {log_path}[/dim]")
        return
    vlog = VerifierLog(log_path)
    try:
        events = list(reversed(vlog.recent(limit=max(1, limit))))
        if not events:
            console.print("[dim](no verifier events)[/dim]")
            return
        rows = load_rows(out) if append and out.exists() else []
        labels = {"a": "accept", "r": "reprompt", "j": "reject"}
        reviewed = 0
        for ev in events:
            ctx_record = vlog.context_for(ev.id)
            ctx = context_from_snapshot(ctx_record)
            rule = RuleBasedVerifier().score(ev.candidate, ctx)
            snippet = ev.candidate if len(ev.candidate) <= 700 else ev.candidate[:700] + "…"
            console.print(
                Panel(
                    snippet,
                    title=(
                        f"{ev.source}  decision={ev.decision} energy={ev.energy:.2f} "
                        f"id={ev.id[:8]}"
                    ),
                    border_style="cyan",
                )
            )
            if ctx_record is not None:
                ctx_preview = render_context_text(ctx_record)
                if ctx_preview:
                    preview = ctx_preview if len(ctx_preview) <= 600 else ctx_preview[:600] + "…"
                    console.print(Panel(preview, title="context", border_style="dim"))
            choice = (
                Prompt.ask(
                    "label [a=accept / r=reprompt / j=reject / s=skip / q=quit]",
                    default=ev.decision[:1].lower(),
                )
                .strip()
                .lower()
            )
            if choice == "q":
                break
            if choice == "s":
                continue
            label = labels.get(choice)
            if label is None:
                console.print("[yellow]skipped (unknown label key)[/yellow]")
                continue
            rows.append(
                VerifierDatasetRow(
                    event_id=f"manual-{ev.id}",
                    source=ev.source,
                    candidate=ev.candidate,
                    ctx_text=render_context_text(ctx_record),
                    ctx_features=feature_dict(
                        ev.candidate, ctx, source=ev.source, rule_verdict=rule
                    ),
                    rule_energy=float(rule.energy),
                    rule_decision=rule.decision,
                    rule_violations=[v.kind for v in rule.violations],
                    label_source="manual_gold",
                    label=label,  # type: ignore[arg-type]
                    reprompt_used=ev.reprompt_used,
                    meta={"event_id": ev.id},
                )
            )
            reviewed += 1
        write_rows(out, rows)
        console.print(f"[green]wrote {len(rows)} row(s)[/green] to {out} (new labels: {reviewed})")
    finally:
        vlog.close()


@verify_app.command("export")
def verify_export(
    out: Path = typer.Option(
        Path("data/verifier_dataset_v1.jsonl"),
        "--out",
        "-o",
        help="Output JSONL dataset path.",
    ),
    gold: Path = typer.Option(
        Path("data/verifier_gold_v1.jsonl"),
        "--gold",
        help="Optional manual gold JSONL path.",
    ),
    limit: int = typer.Option(2000, "--limit", min=1, help="Max telemetry rows to include."),
    mined: bool = typer.Option(True, "--mined/--no-mined", help="Include correction-mined rows."),
    perturb: bool = typer.Option(
        True,
        "--perturb/--no-perturb",
        help="Include synthetic perturbation rows.",
    ),
) -> None:
    """Export unified verifier dataset (telemetry + mined + perturb + optional gold)."""
    settings = get_settings()
    rows = export_verifier_dataset(
        settings.home,
        out_path=out,
        gold_path=gold if gold.exists() else None,
        include_mined=mined,
        include_perturb=perturb,
        limit=limit,
    )
    by_source: dict[str, int] = {}
    for row in rows:
        by_source[row.label_source] = by_source.get(row.label_source, 0) + 1
    source_bits = ", ".join(f"{k}={v}" for k, v in sorted(by_source.items()))
    console.print(f"[green]exported {len(rows)} row(s)[/green] -> {out}")
    if source_bits:
        console.print(f"[dim]{source_bits}[/dim]")


def _score_candidate(
    *,
    text: str,
    source: str,
    explain: bool,
    threshold_reprompt: float,
    threshold_reject: float,
) -> None:
    settings = get_settings()
    fact_store = FactStore(settings.home)
    identity = IdentityStore(settings.home).load()

    # Optional: pull recent corrections from durable memory if it exists.
    corrections: list[MemoryEntry] = []
    try:
        mem_store = _open_memory_store()
    except Exception:
        mem_store = None
    if mem_store is not None:
        try:
            corrections = mem_store.query_by_tag("correction", top_k=5)
        except Exception:
            corrections = []
        finally:
            mem_store.close()

    # Embed-backed normalizer: keeps verifier canonicalization consistent
    # with what the pipeline uses. Falls back to None if it can't init.
    try:
        embedder = get_default_embedder(settings.embed_model)
        normalizer = EmbeddingTermNormalizer(embedder, settings.home)
    except Exception:
        normalizer = None  # type: ignore[assignment]

    ctx = VerifierContext(
        facts=fact_store,
        identity=identity,
        recent_corrections=tuple(corrections),
        normalizer=normalizer,
    )
    v = RuleBasedVerifier(
        threshold_reprompt=threshold_reprompt,
        threshold_reject=threshold_reject,
    )
    verdict = v.score(text, ctx)

    decision_style = {
        "accept": "green",
        "reprompt": "yellow",
        "reject": "red",
    }.get(verdict.decision, "white")
    console.print(
        Panel(
            f"[bold]{verdict.decision}[/bold]  energy={verdict.energy:.2f}  " f"source={source}",
            border_style=decision_style,
        )
    )
    if not verdict.violations:
        console.print("[dim](no violations)[/dim]")
        return
    t = Table(title=f"violations ({len(verdict.violations)})")
    t.add_column("kind")
    t.add_column("weight", justify="right")
    t.add_column("detail", overflow="fold")
    if explain:
        t.add_column("evidence", overflow="fold")
    for vi in verdict.violations:
        row = [vi.kind, f"{vi.weight:.2f}", vi.detail]
        if explain:
            row.append(vi.evidence)
        t.add_row(*row)
    console.print(t)


def _verify_history(home: Path, *, limit: int, explain: bool) -> None:
    log_path = home / "verifier_log.sqlite"
    if not log_path.exists():
        console.print(f"[dim]no verifier log at {log_path}[/dim]")
        return
    vlog = VerifierLog(log_path)
    try:
        events = vlog.recent(limit=max(1, limit))
    finally:
        vlog.close()
    if not events:
        console.print("[dim](no verifier events)[/dim]")
        return
    t = Table(title=f"verifier history ({len(events)})")
    t.add_column("ts", style="dim")
    t.add_column("source")
    t.add_column("decision")
    t.add_column("energy", justify="right")
    t.add_column("reprompt", justify="center")
    t.add_column("candidate", overflow="fold")
    if explain:
        t.add_column("violations", overflow="fold")
    for ev in events:
        from datetime import datetime

        ts_label = datetime.fromtimestamp(ev.ts).strftime("%Y-%m-%d %H:%M:%S")
        row = [
            ts_label,
            ev.source,
            ev.decision,
            f"{ev.energy:.2f}",
            "yes" if ev.reprompt_used else "",
            ev.candidate[:200] + ("…" if len(ev.candidate) > 200 else ""),
        ]
        if explain:
            kinds = ", ".join(v.get("kind", "") for v in ev.violations)
            row.append(kinds)
        t.add_row(*row)
    console.print(t)


@app.command()
def serve(
    host: str | None = typer.Option(None, help="Bind host. Default from LSM_HOST."),
    port: int | None = typer.Option(None, help="Bind port. Default from LSM_PORT."),
    model: str | None = typer.Option(
        None, help="Model tag override (e.g. qwen2.5:7b-instruct-q8_0)."
    ),
    reload: bool = typer.Option(False, help="Autoreload (dev only)."),
    stop: bool = typer.Option(
        False, "--stop", help="Unload the current model from Ollama and exit."
    ),
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
    from lsm.runtime.pipeline_factory import build_default_pipeline

    settings = get_settings()
    effective_model = model or settings.model
    pipe, backend, episodic, _store, memory, outcome_log = build_default_pipeline(
        settings,
        session_id=session_id,
        model=effective_model,
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
        memory.close()
        outcome_log.close()
        episodic.close()


state_app = typer.Typer(name="state", help="Inspect and manage the hidden state s_t.")
state_note_app = typer.Typer(name="note", help="Manage rolling state notes.")
snapshot_app = typer.Typer(
    name="snapshot", help="Create and restore named snapshots of the state directory."
)
identity_app = typer.Typer(
    name="identity",
    help="View and edit the assistant identity profile (pinned identity/role slot).",
)
app.add_typer(state_app, name="state")
state_app.add_typer(state_note_app, name="note")
app.add_typer(snapshot_app, name="snapshot")
fact_app = typer.Typer(name="fact", help="Inspect structured facts (owner + category).")
fact_category_app = typer.Typer(
    name="category",
    help="Register user-approved dynamic fact categories (F.3).",
)
note_app = typer.Typer(
    name="note", help="Manage assistant.notes facts (small thoughts the assistant keeps)."
)
app.add_typer(identity_app, name="identity")
app.add_typer(fact_app, name="fact")
fact_app.add_typer(fact_category_app, name="category")
app.add_typer(note_app, name="note")


@note_app.command("add")
def note_add(
    text: str = typer.Argument(..., help="Free-form note text (small thought)."),
    subject: str | None = typer.Option(
        None, "--subject", help="Optional topic chip the note is about."
    ),
) -> None:
    """Persist a small thought into ``assistant.notes``.

    These appear under the ``assistant notes:`` slot in the s_t projection.
    """
    body = text.strip()
    if not body:
        console.print("Note text is empty.")
        raise typer.Exit(code=2)
    settings = get_settings()
    store = FactStore(settings.home)
    fact = store.upsert(
        Fact(
            owner="assistant",
            category=CAT_ASSISTANT_NOTES,
            subject=subject.strip() if subject else None,
            value=body,
        )
    )
    console.print(f"[green]note added[/green] id={fact.id}")


@note_app.command("list")
def note_list() -> None:
    """List current assistant.notes facts."""
    settings = get_settings()
    store = FactStore(settings.home)
    notes = store.query(owner="assistant", category=CAT_ASSISTANT_NOTES)
    if not notes:
        console.print("[dim](no notes)[/dim]")
        return
    t = Table(title=f"assistant notes ({len(notes)})")
    t.add_column("subject")
    t.add_column("value", overflow="fold")
    for f in notes:
        t.add_row(f.subject or "", f.value)
    console.print(t)


@note_app.command("clear")
def note_clear(
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation."),
) -> None:
    """Delete every assistant.notes fact."""
    settings = get_settings()
    store = FactStore(settings.home)
    if not yes:
        confirmed = Prompt.ask("Delete ALL assistant notes? type 'yes' to confirm", default="no")
        if confirmed.strip().lower() != "yes":
            console.print("aborted")
            raise typer.Exit(code=1)
    n = store.delete_where(category=CAT_ASSISTANT_NOTES)
    console.print(f"[yellow]cleared {n} note(s)[/yellow]")


@fact_category_app.command("add")
def fact_category_add(
    slug: str = typer.Argument(..., help="Short name, e.g. food → user.food"),
    description: str = typer.Option("", "--description", help="Optional description."),
) -> None:
    """Register a dynamic category (``user.<slug>``)."""
    settings = get_settings()
    reg = FactCategoryRegistry(settings.home)
    try:
        defn = reg.register(slug, description=description)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc
    console.print(f"[green]registered[/green] {defn.id}")


@fact_category_app.command("list")
def fact_category_list() -> None:
    """List user-registered dynamic categories."""
    settings = get_settings()
    reg = FactCategoryRegistry(settings.home)
    custom = reg.list_custom()
    if not custom:
        console.print("[dim](no custom categories; built-in categories always available)[/dim]")
        return
    t = Table(title=f"custom categories ({len(custom)})")
    t.add_column("id")
    t.add_column("description", overflow="fold")
    for c in custom:
        t.add_row(c.id, c.description or "")
    console.print(t)


@fact_category_app.command("remove")
def fact_category_remove(
    slug: str = typer.Argument(..., help="Category slug or full id (e.g. food or user.food)."),
) -> None:
    """Remove a user-registered category (does not delete existing facts)."""
    settings = get_settings()
    reg = FactCategoryRegistry(settings.home)
    try:
        removed = reg.remove(slug)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc
    if removed:
        console.print(f"[yellow]removed[/yellow] {FactCategoryRegistry.normalize_slug(slug)}")
    else:
        console.print("[dim]category not found[/dim]")


@fact_app.command("set")
def fact_set(
    category: str = typer.Option(..., "--category", help="Category id, e.g. user.food"),
    value: str = typer.Option(..., "--value", help="Fact value."),
    subject: str | None = typer.Option(None, "--subject", help="Optional subject chip."),
    owner: str = typer.Option("user", "--owner", help="Fact owner (user | assistant)."),
) -> None:
    """Upsert one fact (including custom categories registered via ``fact category add``)."""
    body = value.strip()
    if not body:
        console.print("Value is empty.")
        raise typer.Exit(code=2)
    settings = get_settings()
    reg = FactCategoryRegistry(settings.home)
    store = FactStore(settings.home, category_registry=reg)
    cat = category.strip()
    if not reg.is_allowed(cat):
        console.print(f"Unknown category {cat!r}. Register with: lsm fact category add <slug>")
        raise typer.Exit(code=2)
    if owner not in ("user", "assistant", "shared"):
        console.print("owner must be user, assistant, or shared.")
        raise typer.Exit(code=2)
    fact = store.upsert(
        Fact(
            owner=owner,  # type: ignore[arg-type]
            category=cat,
            subject=subject.strip() if subject else None,
            value=body,
        )
    )
    console.print(f"[green]fact set[/green] id={fact.id} [{fact.category}]")


@fact_app.command("list")
def fact_list(
    owner: str | None = typer.Option(None, "--owner", help="Filter by owner."),
    category: str | None = typer.Option(None, "--category", help="Filter by category."),
) -> None:
    """List structured facts from ``$LSM_HOME/facts.json``."""
    settings = get_settings()
    store = FactStore(settings.home)
    facts = store.query(owner=owner, category=category)
    if not facts:
        console.print("[dim](no facts)[/dim]")
        return
    t = Table(title=f"facts ({len(facts)})")
    t.add_column("category")
    t.add_column("subject")
    t.add_column("value", overflow="fold")
    t.add_column("tags")
    for f in facts:
        t.add_row(
            f.category,
            f.subject or "",
            f.value,
            ", ".join(f.tags) if f.tags else "",
        )
    console.print(t)


@fact_app.command("show")
def fact_show() -> None:
    """Show all structured facts (alias for ``fact list`` with full table)."""
    fact_list(owner=None, category=None)


@fact_app.command("prune")
def fact_prune(
    category: str | None = typer.Option(
        None, "--category", help="Only delete facts in this category."
    ),
    subject: str | None = typer.Option(
        None, "--subject", help="Only delete facts whose subject matches (case-insensitive)."
    ),
    value_contains: str | None = typer.Option(
        None, "--value-contains", help="Only delete facts whose value contains this substring."
    ),
    all_: bool = typer.Option(
        False, "--all", help="Drop every user-owned fact. Keeps assistant.identity."
    ),
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation."),
) -> None:
    """Delete facts in ``$LSM_HOME/facts.json`` matching the given filters.

    Examples:

      lsm fact prune --category user.traits
      lsm fact prune --subject general
      lsm fact prune --value-contains "ice cream"
      lsm fact prune --all
    """
    if not all_ and category is None and subject is None and value_contains is None:
        console.print("Provide --category, --subject, --value-contains, or --all.")
        raise typer.Exit(code=2)
    settings = get_settings()
    store = FactStore(settings.home)
    if not yes:
        target = "ALL user facts" if all_ else "matching facts"
        confirmed = Prompt.ask(f"Delete {target}? type 'yes' to confirm", default="no")
        if confirmed.strip().lower() != "yes":
            console.print("aborted")
            raise typer.Exit(code=1)
    if all_:
        n = store.clear_all(preserve_assistant_identity=True)
    else:
        n = store.delete_where(
            category=category,
            subject=subject,
            value_contains=value_contains,
        )
    console.print(f"[yellow]pruned {n} fact(s)[/yellow]")


@identity_app.command("show")
def identity_show() -> None:
    """Print the assistant name and description used in ``s_t`` identity/role."""
    settings = get_settings()
    profile = IdentityStore(settings.home).load()
    t = Table(show_header=False, box=None)
    t.add_row("name", profile.name)
    t.add_row("description", profile.description)
    console.print(t)
    console.print(
        Panel(
            profile.identity_role_line(),
            title="pinned identity/role line",
            border_style="cyan",
        )
    )


@identity_app.command("set")
def identity_set(
    name: str | None = typer.Option(None, "--name", help="Assistant display name."),
    description: str | None = typer.Option(
        None, "--description", help="What the assistant is / can be."
    ),
) -> None:
    """Update the assistant identity profile (partial updates allowed)."""
    if name is None and description is None:
        console.print("Provide --name and/or --description.")
        raise typer.Exit(code=2)
    settings = get_settings()
    store = IdentityStore(settings.home)
    profile = store.load()
    if name is not None:
        profile.name = name.strip()
    if description is not None:
        profile.description = description.strip()
    store.save(profile)
    console.print("[green]identity updated[/green]")
    console.print(Panel(profile.identity_role_line(), border_style="cyan"))


@state_app.command("show")
def state_show(
    memories: int = typer.Option(
        0,
        "--memories",
        "-m",
        min=0,
        help="If > 0, also show the top-N durable memory hits for s_t.text.",
    ),
) -> None:
    """Print the current state summary and metadata.

    ``s_t`` is a compressed running summary; durable memory holds the
    verbatim-ish atoms. Use ``--memories N`` or ``lsm memory query "..."``
    for ground truth.
    """
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
        console.print(
            "[dim](state is empty \u2014 try: lsm chat, then re-run; or: "
            'lsm memory query "...")[/dim]'
        )

    mem_store = _open_memory_store()
    try:
        try:
            mem_count = mem_store.count()
        except Exception:
            mem_count = None
        if mem_count is not None:
            console.print(f"[dim]memory_count={mem_count}[/dim]")

        if memories > 0:
            if not s.text:
                console.print("[dim](skipping --memories: state is empty, no query text)[/dim]")
            else:
                try:
                    hits = mem_store.query(s.text, top_k=memories)
                except Exception:
                    hits = []
                if not hits:
                    console.print("[dim](no memory hits for s_t.text)[/dim]")
                else:
                    mt = Table(title=f"top {len(hits)} memories for s_t.text")
                    mt.add_column("score", justify="right")
                    mt.add_column("source")
                    mt.add_column("text", overflow="fold")
                    for h in hits:
                        snippet = h.entry.text
                        if len(snippet) > 200:
                            snippet = snippet[:197] + "..."
                        mt.add_row(f"{h.score:.3f}", h.entry.source, snippet)
                    console.print(mt)
    finally:
        mem_store.close()


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


@state_note_app.command("set")
def state_note_set(
    text: str = typer.Argument(..., help="Rolling note text for the state projection."),
) -> None:
    """Set ``State.notes['rolling']``."""
    settings = get_settings()
    store = StateStore(settings.home)
    s = store.load()
    notes = dict(s.notes)
    notes["rolling"] = text.strip()
    from lsm.core.state import State

    store.save(
        State(
            text=s.text,
            vector=s.vector,
            updated_at=s.updated_at,
            revision=s.revision,
            notes=notes,
        )
    )
    console.print("[green]state rolling note updated[/green]")


@state_note_app.command("clear")
def state_note_clear() -> None:
    """Clear ``State.notes['rolling']`` if present."""
    settings = get_settings()
    store = StateStore(settings.home)
    s = store.load()
    notes = dict(s.notes)
    notes.pop("rolling", None)
    from lsm.core.state import State

    store.save(
        State(
            text=s.text,
            vector=s.vector,
            updated_at=s.updated_at,
            revision=s.revision,
            notes=notes,
        )
    )
    console.print("[yellow]state rolling note cleared[/yellow]")


@state_note_app.command("show")
def state_note_show() -> None:
    """Print ``State.notes['rolling']``."""
    settings = get_settings()
    store = StateStore(settings.home)
    s = store.load()
    note = s.notes.get("rolling", "").strip()
    if not note:
        console.print("[dim](no rolling note)[/dim]")
        return
    console.print(Panel(note, title="state.notes[rolling]", border_style="cyan"))


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
    return build_store("lancedb", root=settings.memory_dir / "durable.lance", embedder=embedder)


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
    source: str = typer.Option(None, "--source", help="Source tag. Defaults to each file's path."),
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
    out: Path = typer.Option(Path("experiments"), "--out", "-o", help="Output root directory."),
    run_id: str | None = typer.Option(None, "--run-id"),
) -> None:
    """Run an experiment config and write results under ``experiments/<run_id>/``."""
    from lsm.eval.runner import run_from_config_sync

    run_dir = run_from_config_sync(config_path=config, out_root=out, run_id=run_id)
    console.print(f"[green]results:[/green] {run_dir}")


if __name__ == "__main__":
    app()
