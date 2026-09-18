"""Run events: in-process pub/sub plus an append-only log.

Two consumers, one stream. Subscribers drive live progress (a CLI spinner now,
a streaming UI later); the JSONL sink makes a run replayable after the fact.
Replay matters more than it sounds: when a run produces a surprising verdict,
the event log is the only record of what the agent actually observed before it
decided.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from crucible.core.logging import get_logger

logger = get_logger(__name__)


class EventType(StrEnum):
    """Lifecycle markers emitted by the pipeline."""

    RUN_STARTED = "run.started"
    RUN_FINISHED = "run.finished"
    RUN_FAILED = "run.failed"

    RECON_STARTED = "recon.started"
    RECON_PAGE_VISITED = "recon.page_visited"
    RECON_FINISHED = "recon.finished"

    PLAN_STARTED = "plan.started"
    PLAN_CASE_ADDED = "plan.case_added"
    PLAN_FINISHED = "plan.finished"

    EXEC_STARTED = "exec.started"
    EXEC_CASE_STARTED = "exec.case_started"
    EXEC_CASE_FINISHED = "exec.case_finished"
    EXEC_FINISHED = "exec.finished"

    BROWSER_STARTED = "browser.started"
    BROWSER_PAGE_VISITED = "browser.page_visited"
    BROWSER_FINISHED = "browser.finished"
    BROWSER_UNAVAILABLE = "browser.unavailable"

    #: The interaction lane: pressing, typing into, and choosing from the
    #: controls reconnaissance found. Distinct from the browser walk, which only
    #: looks.
    INTERACT_STARTED = "interact.started"
    INTERACT_ELEMENT_STARTED = "interact.element_started"
    INTERACT_ELEMENT_FINISHED = "interact.element_finished"
    INTERACT_FINISHED = "interact.finished"
    INTERACT_UNAVAILABLE = "interact.unavailable"

    ORACLE_SIGNAL = "oracle.signal"
    VERDICT_REACHED = "oracle.verdict"
    FINDING_RECORDED = "triage.finding"

    QUESTION_ASKED = "question.asked"
    ANSWER_RECEIVED = "question.answered"
    #: A human message that was not a reply to a question. The pipeline answers
    #: it with the run's current progress instead of ignoring it.
    HUMAN_MESSAGE = "human.message"

    BUDGET_WARNING = "budget.warning"


@dataclass(frozen=True, slots=True)
class Event:
    """A single thing that happened during a run."""

    type: EventType
    run_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    ts: datetime = field(default_factory=lambda: datetime.now(UTC))

    def as_dict(self) -> dict[str, Any]:
        return {
            "ts": self.ts.isoformat(),
            "type": self.type.value,
            "run_id": self.run_id,
            **self.payload,
        }


Subscriber = Callable[[Event], Awaitable[None] | None]


class EventBus:
    """Fan-out for run events, with an optional JSONL log.

    A failing subscriber must not abort the run — losing progress output is
    annoying, losing the run is not acceptable — so subscriber exceptions are
    logged and swallowed.
    """

    def __init__(self, run_id: str, *, log_path: Path | None = None) -> None:
        self._run_id = run_id
        self._subscribers: list[Subscriber] = []
        self._log_path = log_path
        self._events: list[Event] = []
        self._lock = asyncio.Lock()

        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def events(self) -> tuple[Event, ...]:
        return tuple(self._events)

    def subscribe(self, subscriber: Subscriber) -> Callable[[], None]:
        """Register a subscriber and return a function that removes it."""
        self._subscribers.append(subscriber)

        def unsubscribe() -> None:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)

        return unsubscribe

    async def emit(self, event_type: EventType, **payload: Any) -> Event:
        """Publish an event to every subscriber and append it to the log."""
        event = Event(type=event_type, run_id=self._run_id, payload=payload)
        self._events.append(event)

        for subscriber in list(self._subscribers):
            try:
                result = subscriber(event)
                if result is not None:
                    await result
            except Exception:
                logger.exception("event_subscriber_failed type=%s", event_type.value)

        if self._log_path is not None:
            async with self._lock:
                try:
                    with self._log_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(event.as_dict(), default=str) + "\n")
                except OSError:
                    # A full disk must not take down an otherwise healthy run.
                    logger.exception("event_log_write_failed path=%s", self._log_path)

        return event

    def counts_by_type(self) -> dict[str, int]:
        """Event counts, useful as a run summary."""
        counts: dict[str, int] = {}
        for event in self._events:
            counts[event.type.value] = counts.get(event.type.value, 0) + 1
        return counts
