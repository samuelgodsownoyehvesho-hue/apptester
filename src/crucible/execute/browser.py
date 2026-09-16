"""The browser lane: look at the application the way a person would.

The check lane talks to the target in text. That is fast, deterministic, and
catches everything that lives on the server -- but it is blind to whatever only
exists once JavaScript has run: a page that renders blank, a button that does
nothing, a form that breaks on submit.

This lane drives a real Chromium browser over the routes reconnaissance already
discovered, and records the screen to video with a screenshot per page. The
video is what makes a finding checkable by someone who is not going to read a
stack trace: "here is the page, here is what it did" is a far stronger argument
than a row in a table.

Deliberately modest. It walks and records; it does not click, type or assert.
Those are per-case actions and belong in a lane of their own.

No browser is bundled with this project, and downloading one may not be
possible on a locked-down network, so the launcher tries Playwright's own
Chromium and then falls back to browsers already installed on the machine.
Chromium, Edge and Chrome share an engine, so any of them renders the target
faithfully.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from crucible.core.events import EventBus, EventType
from crucible.core.logging import get_logger
from crucible.store.artifacts import ArtifactStore

logger = get_logger(__name__)

#: Launch candidates, in order. ``None`` means Playwright's bundled Chromium;
#: the names are Playwright channels for browsers the machine already has.
#: Edge is tried before Chrome because a broken Chrome install is common and
#: fails by exiting silently rather than reporting itself broken.
LAUNCH_CANDIDATES: tuple[str | None, ...] = (None, "msedge", "chrome")

DEFAULT_MAX_PAGES = 8
NAVIGATION_TIMEOUT_MS = 20_000
#: How long to let client-side JavaScript settle after the page loads. Without
#: this the screenshot can catch a skeleton screen instead of the real thing.
SETTLE_TIMEOUT_MS = 5_000

VIEWPORT = {"width": 1024, "height": 720}


class BrowserUnavailable(RuntimeError):
    """No browser could be driven. Distinct from a page that failed to load."""


@dataclass(frozen=True, slots=True)
class Screenshot:
    """One captured page."""

    route: str
    key: str
    status: int


@dataclass(slots=True)
class BrowserRecording:
    """What the browser lane observed and stored."""

    browser: str
    video_key: str | None = None
    screenshots: list[Screenshot] = field(default_factory=list)
    console_errors: list[dict[str, Any]] = field(default_factory=list)
    duration_ms: float = 0.0
    #: Set when the lane ran but could not do its job, e.g. no routes to visit.
    note: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "browser": self.browser,
            "video_key": self.video_key,
            "screenshots": [
                {"route": shot.route, "key": shot.key, "status": shot.status}
                for shot in self.screenshots
            ],
            "console_errors": self.console_errors,
            "duration_ms": round(self.duration_ms, 1),
            "note": self.note,
        }


def _read_video(directory: Path) -> bytes | None:
    """Read the video the browser wrote into ``directory``, if any.

    Synchronous on purpose so it can be handed to a worker thread: a recording
    can run to megabytes, and blocking the event loop to read it would stall
    every other run the dashboard is streaming.
    """
    videos = sorted(directory.glob("*.webm"))
    return videos[0].read_bytes() if videos else None


async def _launch(playwright: Any) -> tuple[Any, str]:
    """Return a launched browser and a label describing it.

    Tries each candidate rather than assuming one exists, because the failure
    mode of a missing or broken browser is an opaque immediate exit. The label
    reaches the report so a finding is never attributed to a browser the reader
    did not expect.
    """
    attempts: list[str] = []
    for channel in LAUNCH_CANDIDATES:
        label = channel or "bundled chromium"
        try:
            if channel is None:
                browser = await playwright.chromium.launch(headless=True)
            else:
                browser = await playwright.chromium.launch(channel=channel, headless=True)
        except Exception as exc:
            attempts.append(f"{label} ({type(exc).__name__})")
            logger.debug("browser_launch_failed candidate=%s", label)
            continue
        logger.info("browser_launched candidate=%s", label)
        return browser, label

    raise BrowserUnavailable(
        "no usable browser; tried " + ", ".join(attempts or ["nothing"])
    )


async def record_walk(
    base_url: str,
    routes: list[str],
    run_id: str,
    artifacts: ArtifactStore,
    *,
    bus: EventBus | None = None,
    max_pages: int = DEFAULT_MAX_PAGES,
) -> BrowserRecording:
    """Walk ``routes`` in a real browser, recording video and screenshots.

    Raises :class:`BrowserUnavailable` only when there is no browser to drive.
    A page that fails to load is an observation, not a reason to abandon the
    walk -- it is recorded and the walk continues.
    """

    async def emit(event_type: EventType, **payload: Any) -> None:
        if bus is not None:
            await bus.emit(event_type, **payload)

    started = time.perf_counter()
    walk = routes[:max_pages]
    if not walk:
        return BrowserRecording(
            browser="none",
            note="reconnaissance discovered no routes to walk",
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )

    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise BrowserUnavailable(f"playwright is not installed: {exc}") from exc

    await emit(EventType.BROWSER_STARTED, base_url=base_url, routes=len(walk))

    work = Path(tempfile.mkdtemp(prefix="crucible-browser-"))
    screenshots: list[Screenshot] = []
    console_errors: list[dict[str, Any]] = []
    video_key: str | None = None
    label = "unknown"

    try:
        async with async_playwright() as playwright:
            browser, label = await _launch(playwright)
            context = await browser.new_context(
                viewport=VIEWPORT,
                record_video_dir=str(work),
                record_video_size=VIEWPORT,
            )
            page = await context.new_page()

            #: Which route a console message belongs to. Messages arrive on a
            #: callback with no page context of their own.
            current = {"route": walk[0]}

            def on_console(message: Any) -> None:
                if message.type == "error":
                    console_errors.append(
                        {
                            "route": current["route"],
                            "kind": "console",
                            "text": str(message.text)[:500],
                        }
                    )

            def on_page_error(error: Any) -> None:
                console_errors.append(
                    {
                        "route": current["route"],
                        "kind": "pageerror",
                        "text": str(error)[:500],
                    }
                )

            page.on("console", on_console)
            page.on("pageerror", on_page_error)

            for route in walk:
                current["route"] = route
                status = 0
                try:
                    response = await page.goto(
                        urljoin(base_url, route),
                        wait_until="load",
                        timeout=NAVIGATION_TIMEOUT_MS,
                    )
                    status = response.status if response is not None else 0
                except Exception as exc:
                    console_errors.append(
                        {
                            "route": route,
                            "kind": "navigation",
                            "text": f"{type(exc).__name__}: {exc}"[:500],
                        }
                    )

                try:
                    await page.wait_for_load_state("networkidle", timeout=SETTLE_TIMEOUT_MS)
                except Exception:
                    logger.debug("browser_settle_timeout route=%s", route)

                png = await page.screenshot()
                ref = artifacts.save_bytes(run_id, "screenshot", png, suffix=".png")
                screenshots.append(Screenshot(route=route, key=ref.key, status=status))
                await emit(
                    EventType.BROWSER_PAGE_VISITED,
                    route=route,
                    status=status,
                    screenshot_key=ref.key,
                )

            # Closing the context is what finalises the video file.
            await context.close()
            await browser.close()

        video_bytes = await asyncio.to_thread(_read_video, work)
        if video_bytes is not None:
            video_key = artifacts.save_bytes(
                run_id, "video", video_bytes, suffix=".webm"
            ).key
    finally:
        shutil.rmtree(work, ignore_errors=True)

    recording = BrowserRecording(
        browser=label,
        video_key=video_key,
        screenshots=screenshots,
        console_errors=console_errors,
        duration_ms=(time.perf_counter() - started) * 1000.0,
    )
    artifacts.save_json(run_id, "browser", recording.as_dict())
    await emit(
        EventType.BROWSER_FINISHED,
        browser=label,
        pages=len(screenshots),
        console_errors=len(console_errors),
        video_key=video_key,
    )
    return recording
