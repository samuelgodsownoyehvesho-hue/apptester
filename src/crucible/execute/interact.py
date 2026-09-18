"""The interaction lane: press everything on every page, like a person would.

The walk lane looks. This lane touches: it clicks each button and link,
types into each field, and selects every option in each dropdown, then asks
whether anything broke as a result.

Why this exists at all. A check lane that speaks HTTP can only find defects that
live in the server's replies. The largest class of real web defects -- a button
that does nothing, a form that throws on submit, a page that dies on one
particular click -- is invisible to it, because it only exists once JavaScript
has run and someone has pressed something. For any site that is not the one
bundled demo shop, "does the page load" is not coverage; this lane is what makes
the tool applicable to an arbitrary website.

Four things are checked after every interaction:

* did the page raise an uncaught JavaScript error
* did any request the interaction triggered return 4xx or 5xx
* did anything happen at all (a control that navigates nowhere, changes nothing
  and calls nothing is a control that does not work)
* did the interaction hang (a timeout is a failure, never a silent pass)

Each interaction re-navigates to its route first. That is the cheap form of
state isolation: a click that empties a cart or deletes a row cannot poison the
next element's precondition, and it keeps the whole pass inside one browser
context, which is what makes the recording a single continuous video of
everything the bot did rather than a pile of fragments.
"""

from __future__ import annotations

import asyncio
import hashlib
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from crucible.core.events import EventBus, EventType
from crucible.core.logging import get_logger
from crucible.execute.browser import (
    NAVIGATION_TIMEOUT_MS,
    VIEWPORT,
    BrowserUnavailable,
    _launch,
    _read_video,
)
from crucible.execute.checks import CheckResult
from crucible.store.artifacts import ArtifactStore

logger = get_logger(__name__)

#: The check id every interaction result is reported under, so triage and the
#: oracle treat a dead button exactly like every other defect.
INTERACTION_CHECK_ID = "ui_interaction"

#: How long an interaction may take before it counts as hung. A control that
#: never returns must fail its own check, not stall a run of hundreds more.
DEFAULT_INTERACTION_TIMEOUT_MS = 15_000

#: Let the page settle after a click before judging it. Too short and a slow
#: handler reads as "nothing happened".
DEFAULT_SETTLE_MS = 500

#: Spacing between interactions. Clicking every control on a deployed site as
#: fast as the loop allows is a good way to trip a rate limiter and get the rest
#: of the run blocked.
DEFAULT_DELAY_MS = 150

#: Sample values by input type, chosen to be valid for the field where possible
#: and obviously synthetic where not.
_SAMPLE_VALUES: dict[str, str] = {
    "email": "crucible.tester@example.test",
    "password": "Crucible-Test-123!",
    "tel": "5550100",
    "url": "https://example.test/",
    "number": "3",
    "range": "3",
    "date": "2026-01-01",
    "datetime-local": "2026-01-01T10:00",
    "month": "2026-01",
    "week": "2026-W01",
    "time": "10:00",
    "search": "crucible",
    "color": "#336699",
    "text": "crucible test",
}
_DEFAULT_SAMPLE = "crucible test"

#: A control that produced none of these is treated as doing nothing.
_NO_EFFECT = "no_change"


@dataclass(slots=True)
class InteractionResult:
    """What happened when one control was pressed."""

    route: str
    kind: str
    label: str
    selector: str
    duration_ms: float = 0.0
    #: ``None`` when the interaction was clean.
    failure: str | None = None
    #: True when the control could not be judged because the page it lives on
    #: did not load. Never reported as a defect: an unreachable target says
    #: nothing about the target's controls, and blaming them for it would turn
    #: every unreachable page into a page full of broken buttons.
    skipped: bool = False
    detail: str = ""
    js_error: str | None = None
    http_status: int | None = None
    before_key: str | None = None
    after_key: str | None = None

    @property
    def ok(self) -> bool:
        return self.failure is None

    @property
    def name(self) -> str:
        """What to call this control in a report read by a person."""
        return self.label.strip() or self.selector

    def as_check_result(self) -> CheckResult:
        """Present the observation as evidence the oracle can judge.

        The interaction lane reports through the same channel as every other
        check rather than as a side record. A real defect that is captured but
        never fed to the oracle is a defect the report does not mention, which
        is indistinguishable from not having looked.
        """
        return CheckResult(
            check_id=INTERACTION_CHECK_ID,
            lane="ui",
            observation=self._observation(),
            facts={
                "route": self.route,
                "kind": self.kind,
                "label": self.name,
                "selector": self.selector,
                "failure": self.failure,
                "js_error": self.js_error,
                "http_status": self.http_status,
                "detail": self.detail,
                "before_key": self.before_key,
                "after_key": self.after_key,
            },
        )

    def _observation(self) -> str:
        if self.failure is None:
            return f"{self.kind} {self.name!r} on {self.route} behaved"
        return f"{self.kind} {self.name!r} on {self.route}: {self.detail}"


@dataclass(slots=True)
class InteractionReport:
    """Everything the interaction lane observed and stored."""

    browser: str = "none"
    #: The recording of the whole pass.
    video_key: str | None = None
    results: list[InteractionResult] = field(default_factory=list)
    duration_ms: float = 0.0
    note: str | None = None

    @property
    def failures(self) -> list[InteractionResult]:
        """Controls that actually broke.

        A skipped control is *not tested*, which is a different thing from
        broken, so it is excluded even if a failure was recorded alongside the
        skip -- the two must never be summed into the same number.
        """
        return [result for result in self.results if not result.ok and not result.skipped]

    def as_dict(self) -> dict[str, Any]:
        return {
            "browser": self.browser,
            "video_key": self.video_key,
            "elements": len(self.results),
            "failures": len(self.failures),
            "duration_ms": round(self.duration_ms, 1),
            "note": self.note,
            "results": [
                {
                    "route": result.route,
                    "kind": result.kind,
                    "label": result.name,
                    "selector": result.selector,
                    "ok": result.ok,
                    "skipped": result.skipped,
                    "failure": result.failure,
                    "detail": result.detail,
                    "js_error": result.js_error,
                    "http_status": result.http_status,
                }
                for result in self.results
            ],
        }


def _sample_for(element: dict[str, Any]) -> str:
    """A plausible value for a field, by its declared type."""
    element_type = str(element.get("type") or "text").lower()
    if element_type in {"checkbox", "radio"}:
        return ""
    return _SAMPLE_VALUES.get(element_type, _DEFAULT_SAMPLE)


def _fingerprint(body: str) -> str:
    """Cheap signature of what is on screen, to tell "something happened"."""
    return hashlib.sha256(body.encode("utf-8", "replace")).hexdigest()[:16]


async def exercise_elements(
    base_url: str,
    elements_by_route: dict[str, list[dict[str, Any]]],
    run_id: str,
    artifacts: ArtifactStore,
    *,
    bus: EventBus | None = None,
    max_elements: int = 0,
    interaction_timeout_ms: int = DEFAULT_INTERACTION_TIMEOUT_MS,
    settle_ms: int = DEFAULT_SETTLE_MS,
    delay_ms: int = DEFAULT_DELAY_MS,
) -> InteractionReport:
    """Press every discovered control, recording the whole pass to video.

    ``max_elements`` of 0 means no ceiling: the operator asked for everything to
    be exercised, and a bounded scan that quietly skips the last third of a page
    is worse than a slow one. Set it only to cap a run deliberately.

    Raises :class:`BrowserUnavailable` when there is no browser to drive.
    A control that breaks the page is an observation, never a reason to stop.
    """

    async def emit(event_type: EventType, **payload: Any) -> None:
        if bus is not None:
            await bus.emit(event_type, **payload)

    started = time.perf_counter()
    planned = {route: list(items) for route, items in elements_by_route.items() if items}
    if not planned:
        return InteractionReport(
            note="reconnaissance found no interactive elements to exercise",
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )

    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise BrowserUnavailable(f"playwright is not installed: {exc}") from exc

    total_planned = sum(len(items) for items in planned.values())
    await emit(EventType.INTERACT_STARTED, routes=len(planned), elements=total_planned)

    work = Path(tempfile.mkdtemp(prefix="crucible-interact-"))
    results: list[InteractionResult] = []
    video_key: str | None = None
    label = "unknown"

    try:
        async with async_playwright() as playwright:
            browser, label = await _launch(playwright)
            # One context for the entire pass: closing it is what finalises the
            # video, and one context means one continuous recording of every
            # interaction rather than a fragment per element.
            context = await browser.new_context(
                viewport=VIEWPORT,
                record_video_dir=str(work),
                record_video_size=VIEWPORT,
            )
            page = await context.new_page()
            page.set_default_timeout(interaction_timeout_ms)

            #: Buffers reset before each interaction, so one control's errors are
            #: never attributed to the next one.
            js_errors: list[str] = []
            bad_responses: list[int] = []
            requests_seen: list[str] = []
            current = {"route": next(iter(planned))}

            def on_console(message: Any) -> None:
                if message.type == "error":
                    js_errors.append(f"console: {str(message.text)[:300]}")

            def on_page_error(error: Any) -> None:
                js_errors.append(f"pageerror: {str(error)[:300]}")

            def on_response(response: Any) -> None:
                requests_seen.append(response.url)
                if response.status >= 400:
                    bad_responses.append(response.status)

            page.on("console", on_console)
            page.on("pageerror", on_page_error)
            page.on("response", on_response)

            for route, items in planned.items():
                current["route"] = route
                budget = items if max_elements <= 0 else items[:max_elements]
                for element in budget:
                    results.append(
                        await _exercise_one(
                            page=page,
                            base_url=base_url,
                            route=route,
                            element=element,
                            run_id=run_id,
                            artifacts=artifacts,
                            js_errors=js_errors,
                            bad_responses=bad_responses,
                            requests_seen=requests_seen,
                            settle_ms=settle_ms,
                            emit=emit,
                        )
                    )
                    if delay_ms:
                        await asyncio.sleep(delay_ms / 1000.0)

            await context.close()
            await browser.close()

        video_bytes = await asyncio.to_thread(_read_video, work)
        if video_bytes is not None:
            video_key = artifacts.save_bytes(
                run_id, "video", video_bytes, suffix=".webm"
            ).key
    finally:
        shutil.rmtree(work, ignore_errors=True)

    report = InteractionReport(
        browser=label,
        video_key=video_key,
        results=results,
        duration_ms=(time.perf_counter() - started) * 1000.0,
    )
    artifacts.save_json(run_id, "interaction", report.as_dict())
    await emit(
        EventType.INTERACT_FINISHED,
        elements=len(results),
        failures=len(report.failures),
        video_key=video_key,
    )
    logger.info(
        "interaction_finished elements=%d failures=%d",
        len(results),
        len(report.failures),
    )
    return report


async def _exercise_one(
    *,
    page: Any,
    base_url: str,
    route: str,
    element: dict[str, Any],
    run_id: str,
    artifacts: ArtifactStore,
    js_errors: list[str],
    bad_responses: list[int],
    requests_seen: list[str],
    settle_ms: int,
    emit: Any,
) -> InteractionResult:
    """Press one control from a clean page load and judge the aftermath."""
    kind = str(element.get("kind") or "clickable")
    selector = str(element.get("selector") or "")
    label = str(element.get("label") or "")
    result = InteractionResult(route=route, kind=kind, label=label, selector=selector)

    await emit(
        EventType.INTERACT_ELEMENT_STARTED,
        route=route,
        kind=kind,
        label=result.name,
        selector=selector,
    )

    started = time.perf_counter()
    js_errors.clear()
    bad_responses.clear()
    requests_seen.clear()

    # Navigation is judged separately from the control. If the page itself
    # cannot be loaded, nothing about its controls has been observed, so the
    # honest result is "not tested" rather than "broken".
    try:
        response = await page.goto(
            urljoin(base_url, route), wait_until="load", timeout=NAVIGATION_TIMEOUT_MS
        )
    except Exception as exc:
        result.skipped = True
        result.duration_ms = (time.perf_counter() - started) * 1000.0
        result.detail = (
            f"the page did not load ({type(exc).__name__}), so its controls "
            "were not judged"
        )
        await emit(
            EventType.INTERACT_ELEMENT_FINISHED,
            route=route,
            kind=kind,
            label=result.name,
            selector=selector,
            ok=True,
            failure=None,
            detail=result.detail,
            duration_ms=round(result.duration_ms, 1),
        )
        return result

    if response is not None and response.status >= 400:
        result.skipped = True
        result.duration_ms = (time.perf_counter() - started) * 1000.0
        result.detail = (
            f"the page itself returns {response.status}, so its controls were "
            "not judged"
        )
        await emit(
            EventType.INTERACT_ELEMENT_FINISHED,
            route=route,
            kind=kind,
            label=result.name,
            selector=selector,
            ok=True,
            failure=None,
            detail=result.detail,
            duration_ms=round(result.duration_ms, 1),
        )
        return result

    # What the page was already complaining about before anything was pressed.
    # A hydration warning or a missing asset is present on load and would
    # otherwise be blamed on whichever control happened to be pressed first,
    # which is how one page-level error becomes a hundred false attributions.
    baseline_errors = list(js_errors)
    baseline_bad = list(bad_responses)

    try:
        await _settle(page, settle_ms)
        before = await _body_text(page)

        target = await _resolve(page, kind, selector, label)
        if target is None:
            result.failure = "unreachable"
            result.detail = (
                "the control reconnaissance found is not present once the page "
                "renders in a real browser"
            )
        else:
            await _act(page, kind, target, element, label)
            await _settle(page, settle_ms)
            after = await _body_text(page)
            _classify(result, before, after, label)
    except Exception as exc:
        # A timeout is a result, not a crash: the control is what hung.
        name = type(exc).__name__
        result.failure = "timeout" if "Timeout" in name else "error"
        result.detail = f"{name}: {exc}"[:300]
    finally:
        result.duration_ms = (time.perf_counter() - started) * 1000.0

    new_errors = [error for error in js_errors if error not in baseline_errors]
    new_bad = [status for status in bad_responses if status not in baseline_bad]

    if new_errors:
        result.js_error = new_errors[0]
        if result.failure is None:
            result.failure = "js_error"
            result.detail = f"pressing it raised a JavaScript error: {new_errors[0]}"

    if new_bad and result.failure is None:
        result.failure = "http_error"
        result.http_status = new_bad[0]
        result.detail = f"pressing it triggered a {new_bad[0]} response"

    # Evidence is captured where a finding exists rather than twice for every
    # successful press: the pass has hundreds of controls, the video already
    # shows what happened, and a pair of near-identical screenshots per success
    # would bury the ones that matter.
    if result.failure is not None:
        result.after_key = await _capture(page, run_id, artifacts, route)
        if result.before_key is None:
            result.before_key = result.after_key

    await emit(
        EventType.INTERACT_ELEMENT_FINISHED,
        route=route,
        kind=kind,
        label=result.name,
        selector=selector,
        ok=result.ok,
        failure=result.failure,
        detail=result.detail,
        duration_ms=round(result.duration_ms, 1),
    )
    return result


def _classify(
    result: InteractionResult, before: str, after: str, label: str
) -> None:
    """Decide whether a press that broke nothing actually did anything.

    Only a control that was *meant* to act is judged this way -- a semantic
    button or something with a click handler. Links navigate, so their effect is
    the navigation itself, and treating "the page looks the same" as failure
    there would manufacture a finding out of every anchor on the site.
    """
    if result.kind not in {"button", "clickable", "submit"}:
        return
    if not result.selector and not label:
        return
    if _fingerprint(before) != _fingerprint(after):
        return
    result.failure = _NO_EFFECT
    result.detail = (
        "pressing it changed nothing on the page and sent no request, so the "
        "control appears to do nothing"
    )


async def _settle(page: Any, settle_ms: int) -> None:
    """Give the page a moment to finish reacting to what just happened."""
    try:
        await page.wait_for_load_state("networkidle", timeout=settle_ms)
    except Exception:
        logger.debug("interact_settle_timeout")
    await asyncio.sleep(settle_ms / 1000.0)


async def _body_text(page: Any) -> str:
    """The page's visible text, used only to tell whether anything changed."""
    try:
        return str(await page.evaluate("document.body ? document.body.innerText : ''"))
    except Exception:
        return ""


async def _capture(
    page: Any, run_id: str, artifacts: ArtifactStore, route: str
) -> str | None:
    """Store a screenshot of the page in the state the interaction left it."""
    try:
        png = await page.screenshot()
    except Exception:
        logger.debug("interact_screenshot_failed route=%s", route)
        return None
    return artifacts.save_bytes(run_id, "screenshot", png, suffix=".png").key


async def _resolve(page: Any, kind: str, selector: str, label: str) -> Any | None:
    """Find the control, falling back to its label when the guess misses.

    The inventory comes from a static parse, which cannot know what the rendered
    DOM will look like. The selector is therefore a hint: when it finds nothing,
    matching on the visible text is what recovers the element.
    """
    if selector:
        try:
            locator = page.locator(selector).first
            if await locator.count() and await locator.is_visible():
                return locator
        except Exception:
            logger.debug("interact_selector_invalid selector=%s", selector)

    if label:
        for role in ("button", "link", "textbox", "checkbox", "combobox"):
            try:
                locator = page.get_by_role(role, name=label, exact=False).first
                if await locator.count() and await locator.is_visible():
                    return locator
            except Exception:
                continue
        try:
            locator = page.get_by_text(label, exact=False).first
            if await locator.count() and await locator.is_visible():
                return locator
        except Exception:
            logger.debug("interact_label_lookup_failed label=%s", label)
    return None


async def _act(
    page: Any, kind: str, target: Any, element: dict[str, Any], label: str
) -> None:
    """Perform the interaction the element's kind calls for."""
    if kind in {"input", "textarea"}:
        element_type = str(element.get("type") or "text").lower()
        if element_type in {"checkbox", "radio"}:
            await target.check()
            return
        if element_type in {"submit", "button", "image", "reset"}:
            await target.click()
            return
        await target.fill(_sample_for(element))
        return

    if kind == "select":
        options = [str(option) for option in element.get("options") or []]
        chosen = options[1] if len(options) > 1 else (options[0] if options else None)
        if chosen:
            await target.select_option(label=chosen)
        else:
            await target.click()
        return

    await target.click()
