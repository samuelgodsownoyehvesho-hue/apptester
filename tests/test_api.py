"""The live control plane: streaming behaviour and refusal paths.

The SSE endpoint cannot be exercised through ``TestClient`` without blocking on
an endless stream, so the generator is driven directly. That is also the better
test: it asserts exactly what a browser receives, in order.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from crucible.api.app import RunState, create_app, stream_events
from crucible.core.config import Settings
from crucible.core.events import Event, EventType
from crucible.store.artifacts import ArtifactStore


def _state(status: str = "running") -> RunState:
    state = RunState(stream_id="live_test", target="http://example.test")
    state.status = status
    return state


@pytest.mark.asyncio
async def test_stream_replays_history_then_signals_the_end() -> None:
    """A viewer attaching late must still receive the whole backlog."""
    state = _state("succeeded")
    state.events.append(
        Event(type=EventType.RECON_STARTED, run_id="live_test", payload={"base_url": "x"})
    )
    state.events.append(
        Event(type=EventType.RUN_FINISHED, run_id="live_test", payload={"decision": "bug"})
    )

    chunks = [chunk async for chunk in stream_events(state)]
    payloads = [
        json.loads(chunk.removeprefix("data: ").strip())
        for chunk in chunks
        if chunk.startswith("data: ")
    ]

    assert [payload["type"] for payload in payloads] == [
        "recon.started",
        "run.finished",
        "stream.end",
    ]


@pytest.mark.asyncio
async def test_stream_on_a_finished_run_terminates() -> None:
    """The stream must not hang on a run that already ended with no events."""
    state = _state("failed")
    chunks = [chunk async for chunk in stream_events(state)]
    assert any("stream.end" in chunk for chunk in chunks)


def test_summary_reports_progress_before_a_result_exists() -> None:
    summary = _state().summary()
    assert summary["status"] == "running"
    assert summary["event_count"] == 0
    # Nothing to cite yet, and an empty list would read as "no problems found".
    assert "citations" not in summary


@pytest.fixture()
def client(tmp_path: Path) -> Iterator[TestClient]:
    settings = Settings(
        crucible_db_url=f"sqlite:///{tmp_path / 'api.db'}",
        crucible_artifacts_dir=tmp_path / "artifacts",
        gemini_api_key="",
        nvidia_api_key="",
        enable_gemini=False,
        enable_nvidia=False,
        enable_ollama=False,
        _env_file=None,
    )
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def test_dashboard_is_served(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Crucible" in response.text


def test_rejects_a_url_that_is_not_http(client: TestClient) -> None:
    """A bare hostname would silently be fetched as a relative path."""
    response = client.post("/api/runs", json={"url": "localhost:3100"})
    assert response.status_code == 400
    assert "http://" in response.json()["detail"]


def test_unknown_run_is_a_404(client: TestClient) -> None:
    assert client.get("/api/runs/live_9999").status_code == 404
    assert client.get("/api/runs/live_9999/events").status_code == 404


def test_unknown_artifact_is_a_404(client: TestClient) -> None:
    assert client.get("/api/artifacts/run_x/video/nope.webm").status_code == 404


def test_stored_artifacts_are_served(tmp_path: Path) -> None:
    """The dashboard plays the video straight out of the artifact store."""
    settings = Settings(
        crucible_db_url=f"sqlite:///{tmp_path / 'api.db'}",
        crucible_artifacts_dir=tmp_path / "artifacts",
        gemini_api_key="",
        nvidia_api_key="",
        enable_gemini=False,
        enable_nvidia=False,
        enable_ollama=False,
        _env_file=None,
    )
    # Written before the app is built, which is also how it happens in a run.
    store = ArtifactStore(tmp_path / "artifacts")
    store.save_text("run_x", "video", "pretend-webm", suffix=".webm")
    key = store.list_run("run_x")[0].key

    with TestClient(create_app(settings)) as client:
        response = client.get(f"/api/artifacts/{key}")

    assert response.status_code == 200
    assert response.content == b"pretend-webm"
    assert response.headers["content-type"].startswith("video/webm")
