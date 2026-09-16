"""The browser lane: graceful degradation, and recording when a browser exists.

The recording test asserts on file *signatures* rather than on file sizes. A
lane that writes a plausible-looking empty file would pass a size check while
producing evidence nobody can watch.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from crucible.execute.browser import BrowserUnavailable, record_walk
from crucible.store.artifacts import ArtifactStore

#: Matroska/WebM container magic. Playwright writes .webm whatever the name.
WEBM_MAGIC = b"\x1a\x45\xdf\xa3"
PNG_MAGIC = b"\x89PNG"


class _QuietHandler(SimpleHTTPRequestHandler):
    """Serves files without writing a log line per request."""

    def log_message(self, *args: object) -> None:
        return


@pytest.fixture()
def live_site(tmp_path: Path) -> Iterator[str]:
    """A real HTTP server, because a browser cannot be driven at a fake one."""
    (tmp_path / "index.html").write_text(
        "<html><head><title>Crucible probe</title></head>"
        "<body><h1>probe</h1></body></html>",
        encoding="utf-8",
    )
    handler = partial(_QuietHandler, directory=str(tmp_path))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture()
def store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(tmp_path / "artifacts")


@pytest.mark.asyncio
async def test_no_routes_means_nothing_to_record_not_a_failure(
    store: ArtifactStore,
) -> None:
    """A target recon found no routes on must not raise from this lane."""
    recording = await record_walk("http://example.test", [], "run_test", store)

    assert recording.video_key is None
    assert recording.screenshots == []
    assert recording.note is not None


@pytest.mark.asyncio
async def test_records_a_real_video_and_screenshot(
    live_site: str, store: ArtifactStore
) -> None:
    try:
        recording = await record_walk(live_site, ["/"], "run_test", store)
    except BrowserUnavailable as exc:
        pytest.skip(f"no usable browser on this machine: {exc}")

    assert recording.video_key is not None, "the walk produced no video"
    video = store.path_for(recording.video_key).read_bytes()
    assert video[:4] == WEBM_MAGIC, "the video file is not a webm container"
    assert len(video) > 1000, "the video is too small to contain a rendered page"

    assert [shot.route for shot in recording.screenshots] == ["/"]
    assert recording.screenshots[0].status == 200
    png = store.path_for(recording.screenshots[0].key).read_bytes()
    assert png[:4] == PNG_MAGIC, "the screenshot is not a png"


@pytest.mark.asyncio
async def test_a_broken_page_is_recorded_and_the_walk_continues(
    live_site: str, store: ArtifactStore
) -> None:
    """A 404 is an observation, not a reason to abandon the walk."""
    try:
        recording = await record_walk(
            live_site, ["/", "/does-not-exist"], "run_test", store
        )
    except BrowserUnavailable as exc:
        pytest.skip(f"no usable browser on this machine: {exc}")

    assert [shot.route for shot in recording.screenshots] == ["/", "/does-not-exist"]
    assert recording.screenshots[1].status == 404
    assert recording.video_key is not None
