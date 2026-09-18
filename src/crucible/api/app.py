"""Crucible's live control plane.

The CLI prints one report when a run finishes. This module exists so a run can
be *watched* while it happens, because reconnaissance against a real
application takes tens of seconds and makes dozens of requests -- and the
difference between "still working" and "wedged" is invisible in a terminal that
says nothing until the end.

Everything here is a thin shell over :func:`crucible.pipeline.run_pipeline`.
The event bus already carries the progress; this module only holds enough state
for a browser to attach to a run that is already under way.

Deliberately in-memory. A single-user development tool does not need a job
queue, and adding one would mean losing runs on restart for no benefit.
"""

from __future__ import annotations

import asyncio
import json
import mimetypes
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session, sessionmaker

from crucible.core.config import Settings
from crucible.core.events import Event, EventBus, EventType
from crucible.core.logging import get_logger
from crucible.core.progress import RunProgress
from crucible.core.qa import ChatChannel
from crucible.llm.chat import ConversationalResponder
from crucible.pipeline import PipelineResult, run_pipeline
from crucible.store.artifacts import ArtifactStore
from crucible.store.db import (
    ensure_parent_dir,
    init_db,
    make_engine,
    make_session_factory,
)
from crucible.store.models import Finding

logger = get_logger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

#: How often a waiting viewer re-checks for new events. The run is a sequence
#: of network probes measured in seconds, so a fifth of a second is
#: indistinguishable from instant and costs nothing.
POLL_INTERVAL_SECONDS = 0.2

#: A run that never terminates must not pin a browser connection open forever.
STREAM_MAX_SECONDS = 30 * 60


class RunRequest(BaseModel):
    """What the browser posts to start a run."""

    url: str


class AnswerRequest(BaseModel):
    """What the browser posts to reply to a question."""

    message_id: str
    text: str


class MessageRequest(BaseModel):
    """A free-form message that is not a reply to any pending question."""

    text: str


#: Sent when someone types at a run that has already stopped. Answering is
#: better than silence: an unanswered message reads as a broken agent.
FINISHED_RUN_REPLY = (
    "That run has already finished, so there is nothing left for me to do "
    "with that. Start another scan and I'll pick it up from there."
)


@dataclass
class RunState:
    """Everything a viewer can know about one run."""

    stream_id: str
    target: str
    status: str = "running"
    events: list[Event] = field(default_factory=list)
    result: PipelineResult | None = None
    error: str | None = None
    #: Held so the task is not garbage collected mid-run.
    task: asyncio.Task[None] | None = None
    #: The live chat channel for this run, if the pipeline has created one.
    channel: ChatChannel | None = None
    #: The plain-language snapshot the agent answers questions from.
    progress: RunProgress | None = None
    #: The reply strategy, held so its provider connections can be released.
    responder: ConversationalResponder | None = None

    def summary(self) -> dict[str, Any]:
        """Flat status payload for the browser."""
        payload: dict[str, Any] = {
            "stream_id": self.stream_id,
            "target": self.target,
            "status": self.status,
            "error": self.error,
            "event_count": len(self.events),
            "conversation": (
                [msg.as_dict() for msg in self.channel.messages]
                if self.channel is not None
                else []
            ),
        }
        if self.result is None:
            return payload

        result = self.result
        payload.update(
            {
                "run_id": result.run_id,
                "target_id": result.target_id,
                "decision": result.decision.value,
                "routes": result.routes,
                "cases": result.cases,
                "executions": result.executions,
                "findings": result.findings,
                "benchmark": result.benchmark.as_dict() if result.benchmark else None,
                "browser": result.browser.as_dict() if result.browser else None,
                "interaction": (
                    result.interaction.as_dict() if result.interaction else None
                ),
                "full_video_key": result.full_video_key,
                "citations": [
                    {
                        "signal": outcome.signal,
                        "check_id": outcome.check_id,
                        "detail": outcome.detail,
                        "confidence": outcome.confidence,
                        "bug_id": outcome.suspected_bug_id,
                    }
                    for outcome in result.outcomes
                    if outcome.violated is True
                ],
            }
        )
        return payload


class RunRegistry:
    """Starts runs and remembers them for the lifetime of the process."""

    def __init__(
        self,
        settings: Settings,
        session_factory: sessionmaker[Session],
        artifacts: ArtifactStore,
    ) -> None:
        self._settings = settings
        self._session_factory = session_factory
        self._artifacts = artifacts
        self._runs: dict[str, RunState] = {}
        self._counter = 0

    @property
    def runs(self) -> tuple[RunState, ...]:
        return tuple(self._runs.values())

    def get(self, stream_id: str) -> RunState | None:
        return self._runs.get(stream_id)

    def start(self, url: str) -> RunState:
        """Begin a run and return immediately, so the browser can subscribe.

        The bus is created here, not inside the pipeline, because the pipeline
        only learns the database run id after reconnaissance has already
        started -- by which point a viewer would have missed the first events.
        """
        self._counter += 1
        stream_id = f"live_{self._counter:04d}"
        state = RunState(stream_id=stream_id, target=url)

        bus = EventBus(stream_id)
        bus.subscribe(state.events.append)
        # The channel is created here so the API can expose it for answers
        # before the pipeline even starts.
        state.channel = ChatChannel(bus)
        # Answering starts here rather than inside the pipeline. Reconnaissance
        # alone takes tens of seconds, and a message typed during it must be
        # queued and answered -- not mistaken for one sent after the run ended.
        state.progress = RunProgress(target=url)
        responder = ConversationalResponder(state.progress, settings=self._settings)
        state.responder = responder
        state.channel.start_responding(responder, aclose=responder.aclose)
        state.task = asyncio.create_task(self._drive(state, bus))
        self._runs[stream_id] = state
        return state

    async def _drive(self, state: RunState, bus: EventBus) -> None:
        """Run the pipeline, converting any failure into a terminal event."""
        try:
            state.result = await run_pipeline(
                state.target,
                self._settings,
                self._session_factory,
                self._artifacts,
                bus=bus,
                channel=state.channel,
                progress=state.progress,
            )
        except Exception as exc:
            state.status = "failed"
            state.error = f"{type(exc).__name__}: {exc}"
            logger.exception("live_run_failed stream=%s", state.stream_id)
            # The viewer is watching the bus, so a crash outside it would leave
            # the stream waiting forever for an event that never comes.
            await bus.emit(EventType.RUN_FAILED, error=state.error)
            return
        finally:
            # The pipeline closes this itself on the normal path. Doing it again
            # here covers a crash, which would otherwise leave a responder loop
            # alive for a run that no longer exists -- still answering the
            # operator as though the scan were still going.
            if state.channel is not None:
                await state.channel.aclose_responding()
        state.status = "succeeded"

    def findings(self, state: RunState) -> list[dict[str, Any]]:
        """Persisted findings for a finished run, newest first."""
        if state.result is None:
            return []
        with self._session_factory() as session:
            rows = (
                session.query(Finding)
                .filter(Finding.run_id == state.result.run_id)
                .all()
            )
        return [
            {
                "title": row.title,
                "severity": str(row.severity),
                "status": str(row.status),
                "bug_id": row.matched_bug_id,
                "cause": row.root_cause,
                "suggested_fix": row.suggested_fix,
                "observed": (row.evidence or {}).get("observed"),
            }
            for row in rows
        ]


def _sse(event: Event) -> str:
    """One server-sent event.

    Sent unnamed on purpose: the event type already travels inside the JSON
    payload, so the browser needs a single ``onmessage`` handler instead of one
    listener per event type -- and new event types reach the UI without a
    client change.
    """
    payload = json.dumps(event.as_dict(), default=str)
    return f"data: {payload}\n\n"


async def stream_events(state: RunState) -> AsyncIterator[str]:
    """Replay a run's history, then follow it until it finishes.

    Polling a list with a cursor beats a per-client queue: a viewer that
    attaches halfway through and one that attaches at the start take the
    identical code path, and there is no window in which an event can be lost
    between "snapshot the backlog" and "start listening".
    """
    cursor = 0
    loop = asyncio.get_running_loop()
    deadline = loop.time() + STREAM_MAX_SECONDS

    while True:
        while cursor < len(state.events):
            yield _sse(state.events[cursor])
            cursor += 1

        if state.status != "running":
            # Terminal marker so the browser can close the connection instead
            # of relying on the server dropping it.
            yield 'data: {"type": "stream.end"}\n\n'
            return

        if loop.time() > deadline:
            yield ": stream timed out\n\n"
            return

        await asyncio.sleep(POLL_INTERVAL_SECONDS)


def create_app(settings: Settings) -> FastAPI:
    """Build the application, wiring the same storage the CLI uses."""
    ensure_parent_dir(settings.crucible_db_url)
    engine = make_engine(settings.crucible_db_url)
    init_db(engine)
    session_factory = make_session_factory(engine)
    artifacts = ArtifactStore(Path(settings.crucible_artifacts_dir))
    registry = RunRegistry(settings, session_factory, artifacts)

    app = FastAPI(title="Crucible", docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))

    @app.post("/api/runs")
    async def start_run(request: RunRequest) -> dict[str, str]:
        url = request.url.strip()
        if not url.startswith(("http://", "https://")):
            raise HTTPException(
                status_code=400, detail="Give a full URL starting with http:// or https://"
            )
        return {"stream_id": registry.start(url).stream_id}

    @app.get("/api/runs")
    async def list_runs() -> list[dict[str, Any]]:
        return [state.summary() for state in registry.runs]

    @app.get("/api/runs/{stream_id}")
    async def run_detail(stream_id: str) -> dict[str, Any]:
        state = registry.get(stream_id)
        if state is None:
            raise HTTPException(status_code=404, detail=f"unknown run {stream_id!r}")
        payload = state.summary()
        payload["finding_details"] = registry.findings(state)
        return payload

    @app.get("/api/artifacts/{key:path}")
    async def artifact(key: str) -> FileResponse:
        """Serve a stored artifact, such as the browser video.

        Resolution goes through the store's own key validation rather than
        joining paths here, so the traversal guard cannot be bypassed by a
        second implementation that forgets it.
        """
        try:
            path = artifacts.path_for(key)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"no artifact {key!r}")
        media_type, _ = mimetypes.guess_type(path.name)
        return FileResponse(path, media_type=media_type or "application/octet-stream")

    @app.get("/api/runs/{stream_id}/events")
    async def run_events(stream_id: str) -> StreamingResponse:
        state = registry.get(stream_id)
        if state is None:
            raise HTTPException(status_code=404, detail=f"unknown run {stream_id!r}")
        return StreamingResponse(
            stream_events(state),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/runs/{stream_id}/answer")
    async def answer_question(stream_id: str, request: AnswerRequest) -> dict[str, Any]:
        """Submit an answer to a question the agent asked."""
        state = registry.get(stream_id)
        if state is None:
            raise HTTPException(status_code=404, detail=f"unknown run {stream_id!r}")
        if state.channel is None:
            raise HTTPException(
                status_code=400, detail="This run has no chat channel"
            )
        accepted = await state.channel.answer(request.message_id, request.text)
        if not accepted:
            raise HTTPException(
                status_code=404,
                detail=f"No pending question with id {request.message_id!r}",
            )
        return {"ok": True}

    @app.post("/api/runs/{stream_id}/message")
    async def post_message(stream_id: str, request: MessageRequest) -> dict[str, Any]:
        """Accept a free-form message, whether or not a question is pending.

        The dashboard's chat box is always enabled, so a typed message usually
        has no question to attach to. Recording it and letting the run's
        responder loop answer it is the whole difference between an agent that
        talks back and one that appears to ignore the operator.
        """
        state = registry.get(stream_id)
        if state is None:
            raise HTTPException(status_code=404, detail=f"unknown run {stream_id!r}")
        if state.channel is None:
            raise HTTPException(status_code=400, detail="This run has no chat channel")

        text = request.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Empty message")

        await state.channel.post(text)
        if state.status != "running" or not state.channel.responding:
            # Nothing is draining the inbox: the run has finished, or it is in
            # the last moments of shutting down. Answer here rather than let the
            # message sit unanswered. It is still recorded, so the conversation
            # stays complete either way.
            await state.channel.inform(FINISHED_RUN_REPLY)
        return {"ok": True}

    return app
