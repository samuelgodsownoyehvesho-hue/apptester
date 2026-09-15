"""Provider clients.

All three providers speak the OpenAI chat-completions protocol, so one client
implementation covers them and a provider swap is a base URL plus a key. The
``openai`` package is imported lazily so that pure-logic modules remain
importable — and testable — without the dependency present.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from crucible.core.config import ProviderConfig
from crucible.core.logging import get_logger

logger = get_logger(__name__)

#: HTTP statuses worth retrying. 429 dominates on free tiers; the rest are
#: transient upstream faults that a backoff usually clears.
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

RETRYABLE_EXCEPTIONS = frozenset(
    {"APIConnectionError", "APITimeoutError", "InternalServerError", "RateLimitError"}
)


def is_retryable(exc: BaseException) -> bool:
    """Return whether an exception is worth retrying.

    Inspects the exception by shape rather than by importing SDK-specific
    classes, so this keeps working across ``openai`` major versions.
    """
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError)):
        return True
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and status in RETRYABLE_STATUS:
        return True
    return type(exc).__name__ in RETRYABLE_EXCEPTIONS


@dataclass(frozen=True, slots=True)
class CompletionResult:
    """A raw completion plus the usage accounting needed for the ledger."""

    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    attempts: int = 1

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class RateLimiter:
    """Sliding-window limiter: at most ``max_per_minute`` acquisitions per 60s.

    Free tiers publish tight per-minute ceilings and reject with 429 well
    before any daily cap. Staying under the ceiling locally is cheaper than
    retrying through it, and it keeps latency predictable.
    """

    def __init__(self, max_per_minute: int, *, window_seconds: float = 60.0) -> None:
        if max_per_minute < 1:
            raise ValueError("max_per_minute must be at least 1")
        self._max = max_per_minute
        self._window = window_seconds
        self._times: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> float:
        """Block until a slot is free. Returns seconds spent waiting."""
        async with self._lock:
            waited = 0.0
            while True:
                now = time.monotonic()
                while self._times and now - self._times[0] >= self._window:
                    self._times.popleft()

                if len(self._times) < self._max:
                    self._times.append(now)
                    return waited

                # Sleep just past the oldest entry's expiry.
                sleep_for = self._window - (now - self._times[0]) + 0.01
                logger.debug("rate_limited provider_wait_s=%.2f", sleep_for)
                await asyncio.sleep(sleep_for)
                waited += sleep_for


class ModelClient:
    """Thin async wrapper over one OpenAI-compatible endpoint."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        rate_limiter: RateLimiter | None = None,
        timeout_seconds: float = 120.0,
        max_attempts: int = 4,
    ) -> None:
        self._config = config
        self._limiter = rate_limiter
        self._timeout = timeout_seconds
        self._max_attempts = max_attempts
        self._client: Any | None = None

    @property
    def config(self) -> ProviderConfig:
        return self._config

    def _get_client(self) -> Any:
        """Lazily construct the SDK client."""
        if self._client is None:
            # Imported here so that importing this module does not require the
            # SDK, and so a missing dependency fails at first use with a clear
            # message rather than at import time with an obscure one.
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:  # pragma: no cover - environment guard
                raise RuntimeError(
                    "The 'openai' package is required for model calls. "
                    "Install dependencies with: uv sync --extra dev"
                ) from exc

            self._client = AsyncOpenAI(
                api_key=self._config.api_key,
                base_url=self._config.base_url,
                timeout=self._timeout,
            )
        return self._client

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        json_mode: bool = False,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> CompletionResult:
        """Issue one chat completion, retrying transient failures."""
        if self._limiter is not None:
            await self._limiter.acquire()

        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        started = time.perf_counter()
        # Counting attempts locally rather than reading them off the retry
        # controller keeps this independent of tenacity's internal types.
        attempt = 0

        async for retry in AsyncRetrying(
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_exponential_jitter(initial=1.0, max=20.0),
            retry=retry_if_exception(is_retryable),
            reraise=True,
        ):
            attempt += 1
            with retry:
                response = await client.chat.completions.create(**kwargs)

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        usage = getattr(response, "usage", None)
        text = response.choices[0].message.content or ""

        return CompletionResult(
            text=text,
            model=getattr(response, "model", self._config.model),
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_ms=elapsed_ms,
            attempts=attempt,
        )

    async def list_models(self) -> list[str]:
        """Return the model ids this endpoint advertises.

        Used to resolve model names empirically instead of hardcoding them.
        Free tiers retire and rename models frequently, so a name read from
        documentation is a guess and a name read from the API is a fact.
        """
        if self._limiter is not None:
            await self._limiter.acquire()

        page = await self._get_client().models.list()
        return sorted(item.id for item in page.data)

    async def aclose(self) -> None:
        """Close the underlying HTTP connection pool."""
        if self._client is not None:
            await self._client.close()
            self._client = None
