"""Event bus behaviour, including the failure modes that matter mid-run."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crucible.core.events import Event, EventBus, EventType


def _entries(path: Path) -> list[str]:
    """List directory entries synchronously.

    Kept out of the async tests deliberately: blocking filesystem calls inside
    a coroutine stall the event loop, and the linter is right to object.
    """
    return sorted(entry.name for entry in path.iterdir())


class TestSubscribers:
    @pytest.mark.asyncio
    async def test_sync_subscriber_receives_events(self) -> None:
        seen: list[Event] = []
        bus = EventBus("run_1")
        bus.subscribe(seen.append)

        await bus.emit(EventType.RUN_STARTED)

        assert len(seen) == 1
        assert seen[0].type is EventType.RUN_STARTED
        assert seen[0].run_id == "run_1"

    @pytest.mark.asyncio
    async def test_async_subscriber_is_awaited(self) -> None:
        seen: list[str] = []

        async def subscriber(event: Event) -> None:
            seen.append(event.type.value)

        bus = EventBus("run_1")
        bus.subscribe(subscriber)
        await bus.emit(EventType.PLAN_STARTED)

        assert seen == ["plan.started"]

    @pytest.mark.asyncio
    async def test_payload_is_carried(self) -> None:
        seen: list[Event] = []
        bus = EventBus("run_1")
        bus.subscribe(seen.append)

        await bus.emit(EventType.RECON_PAGE_VISITED, url="/catalog", status=200)

        assert seen[0].payload == {"url": "/catalog", "status": 200}

    @pytest.mark.asyncio
    async def test_multiple_subscribers_all_receive(self) -> None:
        first: list[Event] = []
        second: list[Event] = []
        bus = EventBus("run_1")
        bus.subscribe(first.append)
        bus.subscribe(second.append)

        await bus.emit(EventType.RUN_FINISHED)

        assert len(first) == 1
        assert len(second) == 1

    @pytest.mark.asyncio
    async def test_unsubscribe_stops_delivery(self) -> None:
        seen: list[Event] = []
        bus = EventBus("run_1")
        unsubscribe = bus.subscribe(seen.append)

        await bus.emit(EventType.RUN_STARTED)
        unsubscribe()
        await bus.emit(EventType.RUN_FINISHED)

        assert len(seen) == 1


class TestFailureIsolation:
    @pytest.mark.asyncio
    async def test_failing_subscriber_does_not_abort_the_emit(self) -> None:
        # Losing progress output is annoying; losing the run is not acceptable.
        healthy: list[Event] = []

        def broken(_event: Event) -> None:
            raise RuntimeError("subscriber exploded")

        bus = EventBus("run_1")
        bus.subscribe(broken)
        bus.subscribe(healthy.append)

        await bus.emit(EventType.RUN_STARTED)

        assert len(healthy) == 1

    @pytest.mark.asyncio
    async def test_failing_async_subscriber_is_contained(self) -> None:
        healthy: list[Event] = []

        async def broken(_event: Event) -> None:
            raise RuntimeError("async subscriber exploded")

        bus = EventBus("run_1")
        bus.subscribe(broken)
        bus.subscribe(healthy.append)

        await bus.emit(EventType.RUN_STARTED)

        assert len(healthy) == 1


class TestEventLog:
    @pytest.mark.asyncio
    async def test_events_are_appended_as_jsonl(self, tmp_path: Path) -> None:
        log = tmp_path / "nested" / "events.jsonl"
        bus = EventBus("run_1", log_path=log)

        await bus.emit(EventType.RUN_STARTED, target="http://localhost:3100")
        await bus.emit(EventType.RUN_FINISHED, findings=3)

        lines = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        assert len(lines) == 2
        assert lines[0]["type"] == "run.started"
        assert lines[0]["target"] == "http://localhost:3100"
        assert lines[1]["findings"] == 3

    @pytest.mark.asyncio
    async def test_log_entries_carry_type_and_run_id(self, tmp_path: Path) -> None:
        bus = EventBus("run_abc", log_path=tmp_path / "e.jsonl")
        await bus.emit(EventType.VERDICT_REACHED, decision="bug")

        entry = json.loads((tmp_path / "e.jsonl").read_text(encoding="utf-8").strip())
        assert entry["run_id"] == "run_abc"
        assert entry["type"] == "oracle.verdict"
        assert "ts" in entry

    @pytest.mark.asyncio
    async def test_no_log_path_writes_nothing(self, tmp_path: Path) -> None:
        bus = EventBus("run_1")
        await bus.emit(EventType.RUN_STARTED)
        assert _entries(tmp_path) == []


class TestReporting:
    @pytest.mark.asyncio
    async def test_counts_by_type(self) -> None:
        bus = EventBus("run_1")
        await bus.emit(EventType.RECON_PAGE_VISITED, url="/")
        await bus.emit(EventType.RECON_PAGE_VISITED, url="/catalog")
        await bus.emit(EventType.RUN_FINISHED)

        counts = bus.counts_by_type()
        assert counts["recon.page_visited"] == 2
        assert counts["run.finished"] == 1

    @pytest.mark.asyncio
    async def test_events_are_retained_in_order(self) -> None:
        bus = EventBus("run_1")
        await bus.emit(EventType.RUN_STARTED)
        await bus.emit(EventType.PLAN_STARTED)

        assert [event.type for event in bus.events] == [
            EventType.RUN_STARTED,
            EventType.PLAN_STARTED,
        ]
