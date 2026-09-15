"""Crucible command line interface.

``doctor`` reports what the routing layer resolves to without touching the
network; ``models`` and ``ping`` make real calls. Keeping the diagnostic
commands separate from the run commands means connectivity and clearance can be
verified before anything expensive starts.
"""

from __future__ import annotations

import asyncio

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
def run(
    target: str = typer.Argument(..., help="URL of the application under test."),
) -> None:
    """Test an application. Not implemented yet; arrives with the recon phase."""
    console.print(
        Panel(
            f"Target: [bold]{target}[/bold]\n\n"
            "The pipeline (recon → plan → execute → oracle) is not built yet.\n"
            "Available today: [bold]doctor[/bold], [bold]models[/bold], [bold]ping[/bold].",
            title="not implemented",
            border_style="yellow",
        )
    )
    raise typer.Exit(code=2)


if __name__ == "__main__":
    app()
