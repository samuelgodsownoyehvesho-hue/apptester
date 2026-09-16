"""HTTP fetching for reconnaissance.

The fetcher is the actual egress point, so the egress allowlist is enforced
here and not only in the caller. A crawler that follows a link off-origin is
performing a request the operator did not authorise, and defence that lives
only at the call site is defence that a future caller can forget.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol
from urllib.parse import urlsplit

import httpx

from crucible.core.logging import get_logger

logger = get_logger(__name__)

DEFAULT_USER_AGENT = "CrucibleBot/0.1 (+internal QA tool)"

#: Responses larger than this are truncated. A test target returning a
#: multi-megabyte page is almost always a mistake, and reading it into memory
#: would cost more than the crawl is worth.
MAX_BODY_BYTES = 3_000_000


class EgressDenied(PermissionError):
    """Raised when a fetch targets a host outside the allowlist."""

    def __init__(self, url: str, reason: str) -> None:
        self.url = url
        super().__init__(f"Egress denied for {url!r}: {reason}")


@dataclass(frozen=True, slots=True)
class FetchResult:
    """The outcome of one HTTP request."""

    url: str
    status: int
    content_type: str = ""
    text: str = ""
    final_url: str = ""
    elapsed_ms: float = 0.0
    truncated: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and 200 <= self.status < 400

    @property
    def is_html(self) -> bool:
        return "html" in self.content_type.lower()

    @property
    def is_json(self) -> bool:
        return "json" in self.content_type.lower()

    def as_dict(self) -> dict[str, object]:
        return {
            "url": self.url,
            "status": self.status,
            "content_type": self.content_type,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "truncated": self.truncated,
            "error": self.error,
        }


class Fetcher(Protocol):
    """Anything that can retrieve a URL."""

    async def fetch(self, url: str) -> FetchResult: ...

    async def aclose(self) -> None: ...


class HttpFetcher:
    """Fetches URLs over HTTP, refusing hosts outside the allowlist."""

    def __init__(
        self,
        *,
        allowed_hosts: frozenset[str] | None = None,
        timeout_seconds: float = 20.0,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> None:
        #: ``None`` means "only the target origin", which the Scout narrows
        #: further. An empty set would deny everything, which is why the two
        #: cases are distinguished.
        self._allowed_hosts = allowed_hosts
        self._client = httpx.AsyncClient(
            timeout=timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": user_agent, "Accept": "text/html,application/json;q=0.9,*/*;q=0.5"},
        )

    def _check_egress(self, url: str) -> None:
        parts = urlsplit(url)
        if parts.scheme not in {"http", "https"}:
            raise EgressDenied(url, f"unsupported scheme {parts.scheme!r}")
        if self._allowed_hosts is not None and parts.netloc.lower() not in self._allowed_hosts:
            raise EgressDenied(url, "host is not in the egress allowlist")

    async def fetch(self, url: str) -> FetchResult:
        """Retrieve ``url``, converting transport failures into a result."""
        self._check_egress(url)
        started = time.perf_counter()

        try:
            response = await self._client.get(url)
        except EgressDenied:
            raise
        except httpx.HTTPError as exc:
            # A failed fetch is data, not a crash: a 404 or a timeout is itself
            # an observation the oracle may need.
            return FetchResult(
                url=url,
                status=0,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                error=f"{type(exc).__name__}: {exc}",
            )

        elapsed = (time.perf_counter() - started) * 1000.0
        raw = response.content
        truncated = len(raw) > MAX_BODY_BYTES
        if truncated:
            raw = raw[:MAX_BODY_BYTES]
            logger.warning("response_truncated url=%s bytes=%d", url, len(raw))

        return FetchResult(
            url=url,
            status=response.status_code,
            content_type=response.headers.get("content-type", ""),
            text=raw.decode(response.encoding or "utf-8", errors="replace"),
            final_url=str(response.url),
            elapsed_ms=elapsed,
            truncated=truncated,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> HttpFetcher:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()


@dataclass
class StaticFetcher:
    """An in-memory fetcher for tests.

    Keys are matched by path, optionally with the query string included. Any
    path not present returns a 404, so a crawl in a test can never silently
    reach the network.
    """

    pages: dict[str, tuple[int, str]] = field(default_factory=dict)
    default_content_type: str = "text/html; charset=utf-8"
    calls: list[str] = field(default_factory=list)

    async def fetch(self, url: str) -> FetchResult:
        self.calls.append(url)
        parts = urlsplit(url)
        # An empty path means the root, not the empty string. Without this the
        # entry page of every crawl would 404.
        path = parts.path or "/"
        key = path
        with_query = f"{path}?{parts.query}" if parts.query else path

        entry = self.pages.get(with_query) or self.pages.get(key)
        if entry is None:
            return FetchResult(url=url, status=404, content_type="text/html", text="<html></html>")

        status, body = entry
        content_type = self.default_content_type
        if path.endswith(".json"):
            content_type = "application/json"
        elif path.endswith(".txt"):
            content_type = "text/plain"

        return FetchResult(url=url, status=status, content_type=content_type, text=body)

    async def aclose(self) -> None:
        return None
