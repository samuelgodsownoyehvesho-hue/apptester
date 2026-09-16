"""The recon Scout: explore a target and build the app map.

The Scout's job is to produce *facts*, not judgements. It records that a
control is unlabelled or a page returned 500; deciding whether either is a
defect belongs to the oracle. Keeping that boundary means the oracle can be
re-run over a stored app map without re-crawling.

Crawling is breadth-first and bounded in three independent ways — page count,
depth, and variants per path — because any one of them alone can be defeated.
A catalog with `?sort=x&page=n` generates unbounded distinct URLs at depth 1,
which is why the per-path variant cap exists.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

from crucible.core.events import EventBus, EventType
from crucible.core.logging import get_logger
from crucible.recon.fetcher import EgressDenied, Fetcher
from crucible.recon.html import PageFacts, extract_page_facts

logger = get_logger(__name__)

DEFAULT_MAX_PAGES = 40
DEFAULT_MAX_DEPTH = 3
#: Distinct query-string variants crawled per path. Without this, faceted
#: search parameters make the frontier grow faster than it drains.
DEFAULT_MAX_VARIANTS_PER_PATH = 3


class ScoutError(RuntimeError):
    """Raised when reconnaissance cannot proceed at all."""


@dataclass(slots=True)
class RouteInfo:
    """What the Scout observed at one URL."""

    url: str
    path: str
    status: int
    depth: int
    title: str | None = None
    content_type: str = ""
    link_count: int = 0
    form_count: int = 0
    unlabelled_control_count: int = 0
    error: str | None = None
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None and 200 <= self.status < 400

    def as_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "path": self.path,
            "status": self.status,
            "depth": self.depth,
            "title": self.title,
            "content_type": self.content_type,
            "link_count": self.link_count,
            "form_count": self.form_count,
            "unlabelled_control_count": self.unlabelled_control_count,
            "truncated": self.truncated,
            "error": self.error,
        }


@dataclass(slots=True)
class AppMapData:
    """The discovered shape of the application under test."""

    base_url: str
    routes: list[RouteInfo] = field(default_factory=list)
    forms: list[dict[str, Any]] = field(default_factory=list)
    unlabelled_controls: list[dict[str, Any]] = field(default_factory=list)
    api_operations: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    crawled_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def route_paths(self) -> list[str]:
        return sorted({route.path for route in self.routes})

    @property
    def failed_routes(self) -> list[RouteInfo]:
        """Routes that returned an error status or failed to fetch.

        A broken internal link shows up here. Whether it is a defect is the
        oracle's call, but it is a fact worth recording either way.
        """
        return [route for route in self.routes if not route.ok]

    def as_dict(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "crawled_at": self.crawled_at.isoformat(),
            "route_count": len(self.routes),
            "routes": [route.as_dict() for route in self.routes],
            "forms": self.forms,
            "unlabelled_controls": self.unlabelled_controls,
            "api_operations": self.api_operations,
            "notes": self.notes,
        }


@dataclass(slots=True)
class RobotsPolicy:
    """A minimal ``robots.txt`` interpretation for a single user agent."""

    disallow: list[str] = field(default_factory=list)

    @classmethod
    async def load(cls, fetcher: Fetcher, base_url: str) -> RobotsPolicy:
        """Fetch and parse ``robots.txt``, tolerating its absence."""
        parts = urlsplit(base_url)
        robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt"
        try:
            result = await fetcher.fetch(robots_url)
        except EgressDenied:
            raise
        except Exception:
            logger.debug("robots_fetch_failed url=%s", robots_url)
            return cls()

        if not result.ok or "text" not in result.content_type:
            return cls()

        disallow: list[str] = []
        applies = False
        for raw in result.text.splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            key, _, value = line.partition(":")
            key, value = key.strip().lower(), value.strip()

            if key == "user-agent":
                applies = value in {"*", "cruciblebot"}
            elif key == "disallow" and applies and value:
                disallow.append(value)

        return cls(disallow=disallow)

    def allows(self, path: str) -> bool:
        """Whether ``path`` may be crawled. Longest matching rule wins."""
        target = path or "/"
        best = -1
        allowed = True
        for pattern in self.disallow:
            if target.startswith(pattern) and len(pattern) > best:
                best = len(pattern)
                allowed = False
        return allowed


class Scout:
    """Breadth-first reconnaissance against one target origin."""

    def __init__(
        self,
        base_url: str,
        fetcher: Fetcher,
        *,
        max_pages: int = DEFAULT_MAX_PAGES,
        max_depth: int = DEFAULT_MAX_DEPTH,
        max_variants_per_path: int = DEFAULT_MAX_VARIANTS_PER_PATH,
        respect_robots: bool = True,
        bus: EventBus | None = None,
        extra_seed_paths: list[str] | None = None,
    ) -> None:
        parts = urlsplit(base_url)
        if parts.scheme not in {"http", "https"} or not parts.netloc:
            raise ScoutError(f"base_url must be an absolute http(s) URL, got {base_url!r}")

        self._base_url = base_url
        self._origin = f"{parts.scheme}://{parts.netloc}"
        self._fetcher = fetcher
        self._max_pages = max_pages
        self._max_depth = max_depth
        self._max_variants = max_variants_per_path
        self._respect_robots = respect_robots
        self._bus = bus
        self._extra_seed_paths = extra_seed_paths or []

    async def _emit(self, event_type: EventType, **payload: Any) -> None:
        if self._bus is not None:
            await self._bus.emit(event_type, **payload)

    async def crawl(self) -> AppMapData:
        """Walk the target and return everything observed."""
        result = AppMapData(base_url=self._base_url)
        await self._emit(EventType.RECON_STARTED, base_url=self._base_url)

        robots = (
            await RobotsPolicy.load(self._fetcher, self._base_url)
            if self._respect_robots
            else RobotsPolicy()
        )
        if robots.disallow:
            result.notes.append(
                f"robots.txt disallows {len(robots.disallow)} path prefix(es); "
                "those paths were not crawled."
            )

        seed = self._base_url
        if not robots.allows(urlsplit(seed).path or "/"):
            raise ScoutError(f"robots.txt disallows the entry point {seed!r}")

        frontier: deque[tuple[str, int]] = deque([(seed, 0)])
        for path in self._extra_seed_paths:
            frontier.append((f"{self._origin}{path}", 0))

        visited: set[str] = set()
        variants: dict[str, int] = {}
        seen_forms: set[str] = set()

        while frontier and len(visited) < self._max_pages:
            url, depth = frontier.popleft()
            if url in visited:
                continue
            visited.add(url)

            path = urlsplit(url).path or "/"
            variants[path] = variants.get(path, 0) + 1

            facts: PageFacts | None = None
            try:
                fetched = await self._fetcher.fetch(url)
            except EgressDenied as exc:
                logger.warning("egress_denied url=%s", url)
                result.routes.append(
                    RouteInfo(
                        url=url,
                        path=path,
                        status=0,
                        depth=depth,
                        error=f"egress denied: {exc}",
                    )
                )
                continue

            route = RouteInfo(
                url=url,
                path=path,
                status=fetched.status,
                depth=depth,
                content_type=fetched.content_type,
                error=fetched.error,
                truncated=fetched.truncated,
            )

            if fetched.is_html and fetched.text:
                facts = extract_page_facts(fetched.text, fetched.final_url or url)
                route.title = facts.title
                route.link_count = len(facts.links)
                route.form_count = len(facts.forms)
                route.unlabelled_control_count = len(facts.unlabelled_controls)

                for form in facts.forms:
                    signature = form.action + "|" + form.method
                    if signature not in seen_forms:
                        seen_forms.add(signature)
                        result.forms.append({"page": path, **form.as_dict()})

                for control in facts.unlabelled_controls:
                    result.unlabelled_controls.append({"page": path, **control.as_dict()})

                if depth < self._max_depth:
                    for link in facts.links:
                        if link in visited:
                            continue
                        link_path = urlsplit(link).path or "/"
                        if variants.get(link_path, 0) >= self._max_variants:
                            continue
                        if not robots.allows(link_path):
                            continue
                        frontier.append((link, depth + 1))

            result.routes.append(route)
            await self._emit(
                EventType.RECON_PAGE_VISITED,
                url=url,
                path=path,
                status=fetched.status,
                depth=depth,
                title=route.title,
            )

        if len(visited) >= self._max_pages and frontier:
            result.notes.append(
                f"Stopped after {self._max_pages} pages with {len(frontier)} URLs "
                "still queued. The map is partial."
            )

        await self._emit(
            EventType.RECON_FINISHED,
            routes=len(result.routes),
            forms=len(result.forms),
            failed=len(result.failed_routes),
        )
        return result

    async def load_openapi(self, spec_url: str, result: AppMapData) -> None:
        """Merge operations from an OpenAPI document into the app map.

        Gives the plan stage real endpoint names rather than only pages, which
        is what makes an API lane possible without guessing at routes.
        """
        fetched = await self._fetcher.fetch(spec_url)
        if not fetched.ok:
            result.notes.append(f"OpenAPI spec unavailable ({fetched.status}): {spec_url}")
            return

        import json

        try:
            document = json.loads(fetched.text)
        except json.JSONDecodeError as exc:
            result.notes.append(f"OpenAPI spec was not valid JSON: {exc}")
            return

        paths = document.get("paths")
        if not isinstance(paths, dict):
            result.notes.append("OpenAPI document has no 'paths' object.")
            return

        for path, operations in paths.items():
            if not isinstance(operations, dict):
                continue
            for method, operation in operations.items():
                if method.lower() not in {"get", "post", "put", "patch", "delete", "head"}:
                    continue
                summary = operation.get("summary") if isinstance(operation, dict) else None
                result.api_operations.append(
                    {
                        "path": path,
                        "method": method.upper(),
                        "summary": summary,
                        "source": spec_url,
                    }
                )

        result.notes.append(
            f"Imported {len(paths)} path(s) and {len(result.api_operations)} "
            f"operation(s) from {spec_url}"
        )
