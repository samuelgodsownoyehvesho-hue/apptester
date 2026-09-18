"""The pipeline: recon, plan, execute, judge, triage, score.

One function per stage so each can be run and tested alone, plus
:func:`run_pipeline` which drives all of them against one target and persists
everything. Verdicts are written per execution: an execution whose probes held
gets ``NOT_A_BUG``, one whose probes violated an invariant gets ``BUG``, and
one that could not run gets ``INSUFFICIENT_EVIDENCE``. Collapsing those three
into a single run-level boolean would destroy exactly the information triage
and the benchmark need.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from crucible.benchmark.score import ScoreReport, fetch_manifest, score
from crucible.core.config import Settings
from crucible.core.events import EventBus, EventType
from crucible.core.logging import get_logger
from crucible.core.progress import FindingNote, ProgressTracker, RunProgress
from crucible.core.qa import ChatChannel
from crucible.execute.browser import (
    BrowserRecording,
    BrowserUnavailable,
    record_walk,
)
from crucible.execute.client import AppClient, HttpAppClient, reset_target
from crucible.execute.interact import InteractionReport, exercise_elements
from crucible.execute.runner import CheckRunner, ExecutionOutcome
from crucible.llm.chat import ConversationalResponder
from crucible.oracle.signals import SIGNALS_BY_CHECK, SignalOutcome, reach_verdict
from crucible.plan.synthesize import synthesize_cases
from crucible.recon.fetcher import HttpFetcher
from crucible.recon.scout import AppMapData, Scout
from crucible.store.artifacts import ArtifactStore
from crucible.store.models import (
    AppMap,
    CaseStatus,
    Execution,
    Finding,
    Run,
    RunStatus,
    Target,
    TestCase,
    TestPlan,
    Verdict,
    VerdictDecision,
    utcnow,
)
from crucible.triage.cluster import write_findings

logger = get_logger(__name__)


def _elements_by_route(app_map: AppMapData) -> dict[str, list[dict[str, Any]]]:
    """Group the element inventory by the route each element was found on."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in app_map.elements:
        page = str(entry.get("page") or "")
        if not page:
            continue
        grouped.setdefault(page, []).append(
            {key: value for key, value in entry.items() if key != "page"}
        )
    return grouped


async def _run_interaction_lane(
    base_url: str,
    app_map: AppMapData,
    run_id: str,
    artifacts: ArtifactStore,
    settings: Settings,
    channel: ChatChannel,
    bus: EventBus,
) -> InteractionReport | None:
    """Press every control reconnaissance found, if there is a browser to do it in.

    Optional in the same way the walk lane is: no browser must degrade the
    report, never fail the run. An empty inventory is reported as a reason
    rather than passing silently, so "no browser findings" is never mistaken for
    "everything worked".
    """
    if not settings.interact_enabled:
        await bus.emit(EventType.INTERACT_UNAVAILABLE, reason="disabled by configuration")
        return None

    elements_by_route = _elements_by_route(app_map)
    total = sum(len(items) for items in elements_by_route.values())
    if not total:
        await bus.emit(
            EventType.INTERACT_UNAVAILABLE,
            reason="reconnaissance found no interactive elements",
        )
        return None

    await channel.inform(
        f"Now pressing every control I found: {total} of them across "
        f"{len(elements_by_route)} page(s). This part takes a while."
    )
    try:
        return await exercise_elements(
            base_url,
            elements_by_route,
            run_id,
            artifacts,
            bus=bus,
            max_elements=settings.interact_max_elements,
            interaction_timeout_ms=settings.interact_timeout_ms,
            settle_ms=settings.interact_settle_ms,
            delay_ms=settings.interact_delay_ms,
        )
    except BrowserUnavailable as exc:
        logger.warning("interaction_lane_unavailable reason=%s", exc)
        await bus.emit(EventType.INTERACT_UNAVAILABLE, reason=str(exc))
        return None
    except Exception as exc:
        # The lane is additive: it must never be the reason a run that has
        # already found things fails to report them.
        logger.exception("interaction_lane_failed")
        await bus.emit(EventType.INTERACT_UNAVAILABLE, reason=f"{type(exc).__name__}: {exc}")
        return None


@asynccontextmanager
async def _client_scope(
    supplied: AppClient | None, base_url: str
) -> AsyncIterator[AppClient]:
    """Yield a client for the execution phase, closing it only if we made it.

    A caller-supplied client is not ours to close: the benchmark stage reads the
    target's manifest through the same object later in the run, and a caller may
    reasonably hold on to it afterwards. Closing it here is what left every check
    talking to a dead transport -- each one errored, and a run that had found ten
    defects reported one while looking perfectly healthy.
    """
    if supplied is not None:
        yield supplied
        return
    async with HttpAppClient(base_url) as client:
        yield client


@dataclass(slots=True)
class PipelineResult:
    """Everything a caller needs to know after one pipeline run."""

    run_id: str
    target_id: str
    decision: VerdictDecision
    routes: int
    cases: int
    executions: int
    findings: int
    benchmark: ScoreReport | None
    app_map: AppMapData
    outcomes: list[SignalOutcome] = field(default_factory=list)
    #: Video and screenshots from the browser lane, when it could run.
    browser: BrowserRecording | None = None
    #: The interaction pass: what was pressed, what broke, and its recording.
    interaction: InteractionReport | None = None
    #: The conversation between the agent and the human operator.
    conversation: list[dict[str, Any]] | None = None

    @property
    def full_video_key(self) -> str | None:
        """The recording of the whole run, preferring the fullest one available.

        The interaction pass drives every page and every control, so its video
        already contains the walk. Falling back to the walk lane's recording
        keeps a video available when interactions were switched off or when no
        elements were found to press.
        """
        if self.interaction is not None and self.interaction.video_key:
            return self.interaction.video_key
        if self.browser is not None:
            return self.browser.video_key
        return None

    def summary(self) -> dict[str, Any]:
        """Flat dict for the CLI report."""
        return {
            "run_id": self.run_id,
            "decision": self.decision.value,
            "routes": self.routes,
            "cases": self.cases,
            "executions": self.executions,
            "findings": self.findings,
            "benchmark": self.benchmark.as_dict() if self.benchmark else None,
            "browser": self.browser.as_dict() if self.browser else None,
            "interaction": self.interaction.as_dict() if self.interaction else None,
            "full_video_key": self.full_video_key,
        }


def persist_recon(
    base_url: str,
    app_map: AppMapData,
    session_factory: sessionmaker[Session],
    artifacts: ArtifactStore,
) -> tuple[str, str]:
    """Persist target, run, and app map rows. Returns ``(target_id, run_id)``."""
    with session_factory() as session:
        target = session.query(Target).filter_by(base_url=base_url).one_or_none()
        if target is None:
            # The demo target serves synthetic data only, so PUBLIC is
            # accurate. A real internal target would be created with
            # data_class="proprietary" and must never route to tier 1.
            target = Target(name=base_url, base_url=base_url, data_class="public")
            session.add(target)
            session.flush()

        run_row = Run(target_id=target.id, status=RunStatus.RUNNING)
        session.add(run_row)
        session.flush()

        session.add(
            AppMap(
                run_id=run_row.id,
                data=app_map.as_dict(),
                route_count=len(app_map.routes),
            )
        )
        artifacts.save_json(run_row.id, "app_map", app_map.as_dict())
        target_id, run_id = target.id, run_row.id
        session.commit()

    return target_id, run_id


async def crawl(
    base_url: str,
    *,
    fetcher: HttpFetcher | None = None,
    max_pages: int = 40,
    bus: EventBus | None = None,
) -> AppMapData:
    """Crawl ``base_url`` and return the app map, without persisting.

    Pass ``bus`` to stream reconnaissance progress; a live viewer is the only
    way to tell a slow crawl from a hung one.
    """
    if fetcher is not None:
        scout = Scout(base_url, fetcher, max_pages=max_pages, bus=bus)
        return await scout.crawl()
    async with HttpFetcher() as real_fetcher:
        scout = Scout(base_url, real_fetcher, max_pages=max_pages, bus=bus)
        return await scout.crawl()


async def run_recon(
    base_url: str,
    settings: Settings,
    session_factory: sessionmaker[Session],
    artifacts: ArtifactStore,
    *,
    fetcher: HttpFetcher | None = None,
    max_pages: int = 40,
    bus: EventBus | None = None,
) -> tuple[str, str, AppMapData]:
    """Crawl the target and persist target, run, and app map.

    Returns ``(target_id, run_id, app_map)``. Pass ``fetcher`` to run against
    a canned transport (tests); otherwise a real HTTP fetcher is used.
    """
    app_map = await crawl(base_url, fetcher=fetcher, max_pages=max_pages, bus=bus)
    target_id, run_id = persist_recon(base_url, app_map, session_factory, artifacts)
    return target_id, run_id, app_map


def _derive_signals(
    outcome: ExecutionOutcome,
) -> list[SignalOutcome]:
    """Run the applicable oracle signals over one check's facts."""
    result = outcome.result
    if result.error is not None:
        return []
    signals = SIGNALS_BY_CHECK.get(result.check_id, ())
    return [signal(result) for signal in signals]


def _persist_verdicts(
    session: Session,
    signals_by_execution: dict[str, list[SignalOutcome]],
    error_by_execution: dict[str, str | None],
) -> dict[str, VerdictDecision]:
    """One verdict per execution, from that execution's signals alone.

    Also sets the execution's final status to match its verdict, so the
    case table reads correctly without joining through verdicts. Returns the
    per-execution decisions for the aggregate.
    """
    decisions: dict[str, VerdictDecision] = {}
    for execution_id, outcomes in signals_by_execution.items():
        violated = [outcome for outcome in outcomes if outcome.violated is True]
        errored = error_by_execution.get(execution_id) is not None

        if violated:
            decision = VerdictDecision.BUG
            confidence = max(outcome.confidence for outcome in violated)
            case_status = CaseStatus.FAILED
        elif errored:
            decision = VerdictDecision.INSUFFICIENT_EVIDENCE
            confidence = 0.0
            case_status = CaseStatus.ERRORED
        elif any(outcome.violated is False for outcome in outcomes):
            decision = VerdictDecision.NOT_A_BUG
            confidence = min(
                (o.confidence for o in outcomes if o.violated is False), default=0.5
            )
            case_status = CaseStatus.PASSED
        else:
            decision = VerdictDecision.INSUFFICIENT_EVIDENCE
            confidence = 0.0
            case_status = CaseStatus.SKIPPED

        decisions[execution_id] = decision
        session.add(
            Verdict(
                execution_id=execution_id,
                decision=decision,
                confidence=confidence,
                signals={"outcomes": [outcome.as_dict() for outcome in outcomes]},
                evidence=[
                    outcome.evidence
                    for outcome in outcomes
                    if outcome.violated is not None
                ],
            )
        )
        session.query(Execution).filter(Execution.id == execution_id).update(
            {Execution.status: case_status}, synchronize_session=False
        )
    return decisions


async def run_pipeline(
    base_url: str,
    settings: Settings,
    session_factory: sessionmaker[Session],
    artifacts: ArtifactStore,
    *,
    fetcher: HttpFetcher | None = None,
    app_client: AppClient | None = None,
    with_benchmark: bool = True,
    with_browser: bool = True,
    bus: EventBus | None = None,
    channel: ChatChannel | None = None,
    progress: RunProgress | None = None,
) -> PipelineResult:
    """Execute the full pipeline against ``base_url``.

    ``fetcher`` and ``app_client`` override the real transports. Tests use
    in-memory doubles so the whole pipeline is exercisable with no network
    and no model provider.

    ``bus`` lets a caller subscribe before the first event fires. A streaming
    viewer needs that: the database run id is not known until reconnaissance
    has already begun, so the caller's own id identifies the stream.
    """
    # The progress snapshot is attached to the bus *before* reconnaissance when
    # a bus already exists: the crawl is the longest silent stretch of a run and
    # therefore the moment an operator is most likely to ask what is going on.
    if progress is None:
        progress = RunProgress(target=base_url)
    if bus is not None:
        bus.subscribe(ProgressTracker(progress))

    target_id, run_id, app_map = await run_recon(
        base_url, settings, session_factory, artifacts, fetcher=fetcher, bus=bus
    )
    if bus is None:
        bus = EventBus(run_id)
        bus.subscribe(ProgressTracker(progress))
    await bus.emit(EventType.RUN_STARTED, base_url=base_url)

    # Interactive chat channel: the pipeline can ask the human operator
    # questions ("I found a login wall — skip it?", "What's the password?")
    # and pauses until they answer via the dashboard.
    if channel is None:
        chat_log_path = artifacts.root / run_id / "conversation.json"
        channel = ChatChannel(bus, log_path=chat_log_path)
        # A run driven straight from the CLI answers questions through the same
        # channel the dashboard uses, rather than falling silent for want of a
        # caller to start the responder.
        cli_responder = ConversationalResponder(progress, settings=settings)
        channel.start_responding(cli_responder, aclose=cli_responder.aclose)
    else:
        # The dashboard has to build the channel before the run has an id, so
        # the conversation log can only be attached now. Without this the
        # conversation lives in memory alone and is lost with the process.
        channel.enable_log(artifacts.root / run_id / "conversation.json")

    # Reconnaissance facts are taken from the app map rather than from the
    # recon events: a caller-supplied bus may already have carried them before
    # this run attached a tracker, and the map is authoritative regardless.
    progress.routes = len(app_map.routes)
    progress.forms = len(app_map.forms)
    progress.broken = len(app_map.failed_routes)
    # A no-op when a responder is already running -- the dashboard starts one its
    # own way -- and the fallback for a caller that supplies a bare channel.
    channel.start_responding(progress.reply)

    await channel.inform(
        f"I'm exploring the site. Found {len(app_map.routes)} pages, "
        f"{len(app_map.forms)} forms so far."
    )

    # ---- plan -------------------------------------------------------------
    await bus.emit(EventType.PLAN_STARTED)
    planned = synthesize_cases(app_map)
    with session_factory() as session:
        plan_row = TestPlan(
            run_id=run_id, strategy="registry_v1", meta={"cases": len(planned)}
        )
        session.add(plan_row)
        session.flush()
        case_rows: list[TestCase] = []
        for item in planned:
            case = TestCase(
                plan_id=plan_row.id,
                title=item.title,
                lane=item.lane,
                requirement_ref=item.requirement_ref,
                risk=item.risk,
            )
            case_rows.append(case)
            session.add(case)
            await bus.emit(
                EventType.PLAN_CASE_ADDED, check_id=item.check_id, risk=item.risk
            )
        session.commit()

    # ---- execute + judge --------------------------------------------------
    signals_by_execution: dict[str, list[SignalOutcome]] = {}
    error_by_execution: dict[str, str | None] = {}
    execution_count = 0

    await channel.inform(
        f"Planning complete. I'm going to run {len(planned)} checks "
        f"to look for bugs."
    )

    # One client spans the reset and every check: previously the reset lived in
    # a block of its own that closed the client before the checks ran.
    async with _client_scope(app_client, base_url) as client:
        # Best-effort reset before probing. Without it a second run inherits the
        # first run's cart, so the same code reports different numbers -- and
        # can reach a different verdict -- purely because of run order.
        await reset_target(client)

        with session_factory() as session:
            runner = CheckRunner(
                client, session, bus, run_id,
                app_map=app_map, questioner=channel,
            )
            outcomes: list[ExecutionOutcome] = await runner.run_all(case_rows)
            for execution_outcome in outcomes:
                execution_count += 1
                error_by_execution[execution_outcome.execution_id] = (
                    execution_outcome.result.error
                )
                signals_by_execution[execution_outcome.execution_id] = _derive_signals(
                    execution_outcome
                )

            decisions = _persist_verdicts(
                session, signals_by_execution, error_by_execution
            )
            session.commit()

    # ---- interaction lane -------------------------------------------------
    # Runs before the verdict is reached so a dead button is judged alongside
    # every other observation. Reported after the fact, it would be evidence the
    # report never mentions -- which is indistinguishable from not having looked.
    interaction = await _run_interaction_lane(
        base_url, app_map, run_id, artifacts, settings, channel, bus
    )
    # Controls on a page that never loaded are excluded: "not tested" is not
    # evidence of a defect, and passing them on would turn an unreachable target
    # into a page full of broken buttons.
    interaction_results = (
        [
            result.as_check_result()
            for result in interaction.results
            if not result.skipped
        ]
        if interaction is not None
        else []
    )

    # reach_verdict is the single aggregation path, shared with the oracle's own
    # tests. It also folds in signals that come from recon rather than from an
    # execution; this block used to rebuild that logic by hand and silently
    # dropped them, so a genuinely unlabelled form control never reached a
    # finding and scored as a miss.
    aggregate, aggregate_confidence, all_outcomes = reach_verdict(
        [outcome.result for outcome in outcomes] + interaction_results, app_map
    )
    _ = decisions  # per-execution; the aggregate above is what the CLI shows

    await bus.emit(
        EventType.VERDICT_REACHED,
        decision=aggregate.value,
        confidence=aggregate_confidence,
    )

    # ---- browser walk -----------------------------------------------------
    await channel.inform(
        f"Checks done. Found {aggregate_confidence:.0%} confidence "
        f"in the verdict. Now opening a real browser to record what the "
        f"site looks like to a visitor."
    )

    browser_recording: BrowserRecording | None = None
    if with_browser:
        # This lane is optional in a way the others are not: it needs a real
        # browser, which may simply not be installable on a locked-down
        # network. A missing browser must degrade the report, never fail the
        # run -- the checks have already produced findings worth reporting.
        try:
            browser_recording = await record_walk(
                base_url, app_map.route_paths, run_id, artifacts, bus=bus
            )
        except BrowserUnavailable as exc:
            logger.warning("browser_lane_unavailable err=%s", exc)
            await bus.emit(EventType.BROWSER_UNAVAILABLE, reason=str(exc))
        except Exception as exc:
            logger.exception("browser_lane_failed")
            await bus.emit(
                EventType.BROWSER_UNAVAILABLE,
                reason=f"{type(exc).__name__}: {exc}",
            )

    # ---- triage -----------------------------------------------------------
    await channel.inform(
        "Browser recording done. Now grouping findings and checking "
        "accuracy against known defects."
    )

    with session_factory() as session:
        findings: list[Finding] = await write_findings(run_id, all_outcomes, bus)
        session.add_all(findings)
        session.commit()
        # Triage has now produced the plain-language cause and fix, which the
        # event stream does not carry. Folding them in lets the agent answer
        # "why" with the real reason instead of just the finding's title.
        progress.replace_notes(
            [
                FindingNote(
                    title=row.title,
                    cause=row.root_cause or "",
                    fix=row.suggested_fix or "",
                )
                for row in findings
            ]
        )

    # ---- benchmark --------------------------------------------------------
    benchmark_report: ScoreReport | None = None
    if with_benchmark:
        # Benchmarking is a harness concern, not part of testing the target. A
        # real application will not expose a ground-truth manifest, and losing
        # an otherwise complete run because that optional input is missing
        # would be the wrong trade -- so the fetch is inside the guard too.
        try:
            if app_client is not None:
                manifest = await fetch_manifest(app_client)
            else:
                async with HttpAppClient(base_url) as bench_client:
                    manifest = await fetch_manifest(bench_client)
            benchmark_report = score(manifest, findings, all_outcomes)
        except ValueError as exc:
            logger.warning("benchmark_unavailable err=%s", exc)
        else:
            artifacts.save_json(run_id, "benchmark", benchmark_report.as_dict())

    if benchmark_report:
        await channel.inform(
            f"Scan complete! Found {len(findings)} bugs. "
            f"Accuracy: {benchmark_report.recall:.0%} of known defects found, "
            f"{benchmark_report.precision:.0%} precision (no false alarms)."
        )
    else:
        await channel.inform(
            f"Scan complete! Found {len(findings)} issue(s) "
            f"across {len(app_map.routes)} pages."
        )

    with session_factory() as session:
        run_row = session.get(Run, run_id)
        if run_row is not None:
            run_row.status = RunStatus.SUCCEEDED
            run_row.finished_at = utcnow()
            session.commit()

    await bus.emit(
        EventType.RUN_FINISHED,
        decision=aggregate.value,
        findings=len(findings),
        benchmark=benchmark_report.as_dict() if benchmark_report else None,
    )

    # Stop answering only now. The stop sentinel queues behind anything the
    # operator already sent, so every message they typed still gets a reply, but
    # nothing can be appended after the conversation has been written to disk --
    # which is what previously left the log missing its own last messages.
    await channel.aclose_responding()
    channel.save()

    return PipelineResult(
        run_id=run_id,
        target_id=target_id,
        decision=aggregate,
        routes=len(app_map.routes),
        cases=len(case_rows),
        executions=execution_count,
        findings=len(findings),
        benchmark=benchmark_report,
        app_map=app_map,
        outcomes=all_outcomes,
        browser=browser_recording,
        interaction=interaction,
        conversation=[msg.as_dict() for msg in channel.messages],
    )
