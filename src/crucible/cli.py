"""Crucible command line interface.

``doctor`` reports what the routing layer resolves to without touching the
network; ``models`` and ``ping`` make real calls. Keeping the diagnostic
commands separate from the run commands means connectivity and clearance can be
verified before anything expensive starts.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from crucible import __version__
from crucible.core.budget import BudgetExceeded, RunBudget
from crucible.core.config import Settings, get_settings
from crucible.core.logging import configure_logging
from crucible.llm.ledger import CostLedger
from crucible.llm.providers import ModelClient
from crucible.llm.router import ModelRouter, NoProviderAvailable, Tier
from crucible.llm.sensitivity import DataClass, ProviderName, SensitivityViolation

if TYPE_CHECKING:
    from crucible.pipeline import PipelineResult

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Crucible — autonomous multi-agent QA platform.",
)
console = Console()


def _bootstrap() -> tuple[Settings, ModelRouter, CostLedger]:
    """Load settings, configure logging, and build a router with a budget."""
    settings = get_settings()
    configure_logging(settings.log_level, json_output=settings.log_json)
    ledger = CostLedger()
    budget = RunBudget(settings.max_run_cost_usd, settings.max_run_tokens)
    router = ModelRouter(settings, ledger, budget=budget, run_id="cli")
    return settings, router, ledger


@app.command()
def version() -> None:
    """Print the version."""
    console.print(f"crucible {__version__}")


@app.command()
def doctor() -> None:
    """Report provider configuration and per-tier routing, without network calls.

    This is the command to run when something is not working. It answers the
    question "which provider would this call actually go to?" for every tier
    and data class, which is not always obvious from the .env file alone.
    """
    settings, router, _ = _bootstrap()

    console.print(Panel(f"crucible {__version__}", title="doctor", expand=False))

    providers = Table(title="Providers", show_lines=False)
    providers.add_column("Provider")
    providers.add_column("Tier")
    providers.add_column("Configured")
    providers.add_column("Model")
    providers.add_column("Cleared for")

    from crucible.llm.sensitivity import CLEARANCES

    tier_of = {
        ProviderName.GEMINI: "1 (cheap)",
        ProviderName.NVIDIA: "2 (frontier)",
        ProviderName.OLLAMA: "3 (local)",
    }
    for name in ProviderName:
        config = settings.provider_config(name)
        cleared = ", ".join(sorted(c.value for c in CLEARANCES.get(name, frozenset())))
        providers.add_row(
            name.value,
            tier_of[name],
            "[green]yes[/green]" if config else "[dim]no[/dim]",
            config.model if config else "[dim](none)[/dim]",
            cleared or "[dim]nothing[/dim]",
        )
    console.print(providers)

    matrix = Table(title="Routing by tier and data class")
    matrix.add_column("Tier")
    matrix.add_column("PUBLIC")
    matrix.add_column("PROPRIETARY")
    report = router.diagnostics()
    for tier in Tier:
        row = report["tiers"][tier.value]
        public = row["public"]
        proprietary = row["proprietary"]
        matrix.add_row(
            tier.value,
            public if public != "unavailable" else "[red]unavailable[/red]",
            proprietary if proprietary != "unavailable" else "[red]unavailable[/red]",
        )
    console.print(matrix)

    console.print(
        "[dim]Escalated[/dim] means a configured provider was passed over for "
        "clearance reasons and a more constrained one was used instead."
    )

    budget = RunBudget(settings.max_run_cost_usd, settings.max_run_tokens)
    console.print(
        f"\nBudget per run: [bold]${budget.max_cost_usd:.2f}[/bold], "
        f"[bold]{budget.max_tokens:,}[/bold] tokens, "
        f"{settings.max_requests_per_minute} req/min"
    )
    if not settings.available_providers():
        console.print(f"[red]{settings.missing_provider_hint()}[/red]")


@app.command()
def models(
    provider: str = typer.Argument("gemini", help="Provider to query."),
) -> None:
    """List the models a configured provider actually advertises.

    Free tiers retire and rename models often enough that a name taken from
    documentation is a guess. This reads the real list from the endpoint.
    """
    settings, _, _ = _bootstrap()

    try:
        name = ProviderName(provider.lower())
    except ValueError:
        valid = ", ".join(p.value for p in ProviderName)
        console.print(f"[red]Unknown provider {provider!r}. Valid: {valid}[/red]")
        raise typer.Exit(code=2) from None

    config = settings.provider_config(name)
    if config is None:
        console.print(f"[red]Provider {name.value!r} is not configured.[/red]")
        raise typer.Exit(code=2)

    async def _list() -> list[str]:
        client = ModelClient(config)
        try:
            return await client.list_models()
        finally:
            await client.aclose()

    try:
        available = asyncio.run(_list())
    except Exception as exc:
        console.print(f"[red]Failed to list models: {type(exc).__name__}: {exc}[/red]")
        raise typer.Exit(code=1) from None

    console.print(f"[bold]{len(available)}[/bold] models from {name.value}:")
    for model_id in available:
        marker = " [green]<- configured[/green]" if model_id == config.model else ""
        console.print(f"  {model_id}{marker}")


@app.command()
def ping() -> None:
    """Make one real call per tier to verify connectivity and clearance.

    Exercises the full guard chain: provider selection, clearance check,
    budget pre-flight, the request itself, and ledger recording.
    """
    _, router, ledger = _bootstrap()

    async def _run() -> None:
        results = Table(title="Live calls")
        results.add_column("Request")
        results.add_column("Routed to")
        results.add_column("Result")

        # 1. Tier 1 with public data: the expected happy path.
        try:
            reply = await router.complete(
                tier=Tier.CHEAP,
                data_class=DataClass.PUBLIC,
                agent="doctor",
                messages=[{"role": "user", "content": "Reply with exactly: OK"}],
                max_tokens=16,
            )
            preview = reply.text.strip()[:40] or "(empty)"
            results.add_row(
                "cheap + PUBLIC",
                "gemini",
                f"[green]{preview}[/green] ({reply.total_tokens} tok, "
                f"{reply.latency_ms:.0f}ms)",
            )
        except Exception as exc:
            results.add_row("cheap + PUBLIC", "gemini", f"[red]{type(exc).__name__}[/red]")

        # 2. Proprietary data on the cheap tier: must not reach Gemini.
        try:
            router.select(Tier.CHEAP, DataClass.PROPRIETARY)
            results.add_row(
                "cheap + PROPRIETARY", "-", "[dim]routed[/dim]"
            )
        except NoProviderAvailable:
            results.add_row(
                "cheap + PROPRIETARY",
                "-",
                "[yellow]refused (no cleared provider)[/yellow]",
            )

        # 3. The judge tier: must never silently downgrade to tier 1.
        try:
            router.select(Tier.FRONTIER, DataClass.PUBLIC)
            results.add_row("frontier + PUBLIC", "-", "[dim]routed[/dim]")
        except NoProviderAvailable:
            results.add_row(
                "frontier + PUBLIC",
                "—",
                "[yellow]refused (will not downgrade)[/yellow]",
            )

        # 4. Secrets: refused everywhere, always.
        try:
            router.select(Tier.CHEAP, DataClass.SECRET)
            results.add_row("ANY + SECRET", "-", "[red]ALLOWED - BUG[/red]")
        except SensitivityViolation:
            results.add_row("ANY + SECRET", "-", "[green]refused[/green]")

        console.print(results)
        await router.aclose()

    try:
        asyncio.run(_run())
    except BudgetExceeded as exc:
        console.print(f"[red]Budget exceeded: {exc}[/red]")
        raise typer.Exit(code=1) from None

    console.print(f"\n[bold]Ledger[/bold]: {ledger.summary()}")


@app.command()
def recon(
    target: str = typer.Argument(..., help="URL of the application under test."),
    max_pages: int = typer.Option(40, "--max-pages", help="Crawl page limit."),
    db: str | None = typer.Option(None, "--db", help="Override the database URL."),
) -> None:
    """Crawl a target, build the app map, and persist it."""
    settings, _, _ = _bootstrap()
    if db:
        object.__setattr__(settings, "crucible_db_url", db)

    from pathlib import Path

    from crucible.pipeline import crawl, persist_recon
    from crucible.store.artifacts import ArtifactStore
    from crucible.store.db import (
        ensure_parent_dir,
        init_db,
        make_engine,
        make_session_factory,
    )

    db_url = settings.crucible_db_url
    ensure_parent_dir(db_url)
    engine = make_engine(db_url)
    init_db(engine)
    session_factory = make_session_factory(engine)
    artifacts = ArtifactStore(Path(settings.crucible_artifacts_dir))

    async def _crawl() -> tuple[str, str, int, int, int, list[str]]:
        app_map = await crawl(target, max_pages=max_pages)
        target_id, run_id = persist_recon(target, app_map, session_factory, artifacts)
        return (
            target_id,
            run_id,
            len(app_map.routes),
            len(app_map.forms),
            len(app_map.unlabelled_controls),
            app_map.notes,
        )

    try:
        target_id, run_id, routes, forms, unlabelled, notes = asyncio.run(_crawl())
    except Exception as exc:
        console.print(f"[red]Recon failed: {type(exc).__name__}: {exc}[/red]")
        raise typer.Exit(code=1) from None

    table = Table(title="App map")
    table.add_column("Metric")
    table.add_column("Count", justify="right")
    table.add_row("Routes discovered", str(routes))
    table.add_row("Forms", str(forms))
    table.add_row("Unlabelled controls", str(unlabelled))
    console.print(table)
    for note in notes:
        console.print(f"[dim]note: {note}[/dim]")
    console.print(
        f"[green]Persisted[/green] target [bold]{target_id}[/bold], "
        f"run [bold]{run_id}[/bold]"
    )


@app.command()
def run(
    target: str = typer.Argument(..., help="URL of the application under test."),
    db: str | None = typer.Option(None, "--db", help="Override the database URL."),
    no_benchmark: bool = typer.Option(False, "--no-benchmark", help="Skip scoring."),
) -> None:
    """Run the full pipeline: recon, plan, execute, judge, triage, score."""
    settings, _, _ = _bootstrap()
    if db:
        object.__setattr__(settings, "crucible_db_url", db)

    from pathlib import Path

    from crucible.pipeline import run_pipeline
    from crucible.store.artifacts import ArtifactStore
    from crucible.store.db import (
        ensure_parent_dir,
        init_db,
        make_engine,
        make_session_factory,
    )
    from crucible.store.models import VerdictDecision

    db_url = settings.crucible_db_url
    ensure_parent_dir(db_url)
    engine = make_engine(db_url)
    init_db(engine)
    session_factory = make_session_factory(engine)
    artifacts = ArtifactStore(Path(settings.crucible_artifacts_dir))

    async def _execute() -> PipelineResult:
        return await run_pipeline(
            target,
            settings,
            session_factory,
            artifacts,
            with_benchmark=not no_benchmark,
        )

    try:
        result: PipelineResult = asyncio.run(_execute())
    except Exception as exc:
        console.print(f"[red]Run failed: {type(exc).__name__}: {exc}[/red]")
        raise typer.Exit(code=1) from None

    decision_style = {
        VerdictDecision.BUG: "[bold red]BUG[/bold red]",
        VerdictDecision.NOT_A_BUG: "[bold green]NOT_A_BUG[/bold green]",
        VerdictDecision.INSUFFICIENT_EVIDENCE: (
            "[bold yellow]INSUFFICIENT_EVIDENCE[/bold yellow]"
        ),
    }[result.decision]

    console.print(
        Panel(
            f"Run [bold]{result.run_id}[/bold] against {target}\n"
            f"Verdict: {decision_style}",
            title="pipeline result",
            expand=False,
        )
    )

    stages = Table(title="Stage summary")
    stages.add_column("Stage")
    stages.add_column("Result")
    stages.add_row("Recon", f"{result.routes} routes")
    stages.add_row("Plan", f"{result.cases} cases")
    stages.add_row("Execute", f"{result.executions} executions")
    stages.add_row("Triage", f"{result.findings} finding(s)")
    console.print(stages)

    violated = [outcome for outcome in result.outcomes if outcome.violated is True]
    if violated:
        detail = Table(title="Invariant violations (evidence)")
        detail.add_column("Signal")
        detail.add_column("Detail")
        for outcome in violated:
            detail.add_row(outcome.signal, outcome.detail)
        console.print(detail)

    benchmark = result.benchmark
    if benchmark is not None:
        score_table = Table(title="Benchmark (vs declared defects)")
        score_table.add_column("Metric")
        score_table.add_column("Value", justify="right")
        score_table.add_row("Declared active", str(benchmark.declared_active))
        score_table.add_row("Found", str(benchmark.found))
        score_table.add_row("Recall", f"{benchmark.recall:.0%}")
        score_table.add_row("Precision", f"{benchmark.precision:.0%}")
        if benchmark.missed:
            score_table.add_row("Missed", ", ".join(benchmark.missed))
        if benchmark.false_positives:
            score_table.add_row(
                "False positives", ", ".join(benchmark.false_positives)
            )
        console.print(score_table)


if __name__ == "__main__":
    app()
