"""The application-facing HTTP client.

Checks speak to the target through :class:`AppClient`, never through ``httpx``
directly. The indirection is what makes the whole oracle layer testable
offline: tests supply an in-memory client that simulates the target, and the
same checks run unmodified against a real server in production.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlsplit

import httpx

from crucible.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ApiResponse:
    """The part of a response a check can reason about."""

    status: int
    is_json: bool = False
    json_body: Any = None
    text: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and 200 <= self.status < 400


class AppClient(Protocol):
    """What a check needs from the application under test."""

    async def get(
        self, path: str, *, params: dict[str, str] | None = None
    ) -> ApiResponse: ...

    async def post(self, path: str, json_body: Any) -> ApiResponse: ...

    async def delete(
        self, path: str, *, params: dict[str, str] | None = None
    ) -> ApiResponse: ...

    async def aclose(self) -> None: ...

    async def __aenter__(self) -> AppClient: ...

    async def __aexit__(self, *exc: object) -> None: ...


class HttpAppClient:
    """Talks to one origin only. The origin is fixed at construction."""

    def __init__(self, base_url: str, *, timeout_seconds: float = 20.0) -> None:
        parts = urlsplit(base_url)
        if parts.scheme not in {"http", "https"} or not parts.netloc:
            raise ValueError(f"base_url must be absolute, got {base_url!r}")
        self._origin = f"{parts.scheme}://{parts.netloc}"
        self._client = httpx.AsyncClient(
            base_url=self._origin, timeout=timeout_seconds
        )

    async def _send(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, str] | None = None,
        json_body: Any = None,
    ) -> ApiResponse:
        try:
            response = await self._client.request(
                method, path, params=params, json=json_body
            )
        except httpx.HTTPError as exc:
            return ApiResponse(status=0, error=f"{type(exc).__name__}: {exc}")

        content_type = response.headers.get("content-type", "")
        is_json = "json" in content_type.lower()
        body: Any = None
        if is_json:
            try:
                body = response.json()
            except ValueError:
                is_json = False
        return ApiResponse(
            status=response.status_code,
            is_json=is_json,
            json_body=body,
            text=response.text[:4000],
        )

    async def get(
        self, path: str, *, params: dict[str, str] | None = None
    ) -> ApiResponse:
        return await self._send("GET", path, params=params)

    async def post(self, path: str, json_body: Any) -> ApiResponse:
        return await self._send("POST", path, json_body=json_body)

    async def delete(
        self, path: str, *, params: dict[str, str] | None = None
    ) -> ApiResponse:
        return await self._send("DELETE", path, params=params)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> HttpAppClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()
