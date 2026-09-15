"""Provider plumbing that does not require a network.

The completion path itself needs a live endpoint and is exercised by the
integration tests once a provider key is configured. These tests cover the
decision logic that sits in front of it, where a mistake shows up as either a
429 storm or a stalled run.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from crucible.llm.providers import RETRYABLE_STATUS, RateLimiter, is_retryable


class FakeAPIError(Exception):
    """Stands in for an SDK error carrying an HTTP status."""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(f"status {status_code}")


class TestRetryClassification:
    @pytest.mark.parametrize("status", sorted(RETRYABLE_STATUS))
    def test_retryable_statuses(self, status: int) -> None:
        assert is_retryable(FakeAPIError(status))

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
    def test_client_errors_are_not_retried(self, status: int) -> None:
        # Retrying a 401 burns budget and hides a configuration mistake.
        assert not is_retryable(FakeAPIError(status))

    def test_transport_failures_are_retried(self) -> None:
        assert is_retryable(TimeoutError())
        assert is_retryable(ConnectionError())

    @pytest.mark.parametrize(
        "name",
        ["APIConnectionError", "APITimeoutError", "InternalServerError", "RateLimitError"],
    )
    def test_sdk_exceptions_match_by_name(self, name: str) -> None:
        # Matched by class name so this keeps working across SDK major versions.
        exc = type(name, (Exception,), {})()
        assert is_retryable(exc)

    def test_unrelated_errors_are_not_retried(self) -> None:
        assert not is_retryable(ValueError("bad payload"))
        assert not is_retryable(KeyError("missing"))


class TestRateLimiter:
    def test_rejects_nonsensical_limit(self) -> None:
        with pytest.raises(ValueError):
            RateLimiter(0)

    @pytest.mark.asyncio
    async def test_allows_burst_within_limit(self) -> None:
        limiter = RateLimiter(3, window_seconds=5.0)
        started = time.monotonic()
        for _ in range(3):
            await limiter.acquire()
        # A burst inside the allowance must not block.
        assert time.monotonic() - started < 0.5

    @pytest.mark.asyncio
    async def test_blocks_once_window_is_full(self) -> None:
        limiter = RateLimiter(2, window_seconds=0.3)
        await limiter.acquire()
        await limiter.acquire()

        waited = await limiter.acquire()
        assert waited > 0.0

    @pytest.mark.asyncio
    async def test_slot_is_reusable_after_window_expires(self) -> None:
        limiter = RateLimiter(1, window_seconds=0.2)
        await limiter.acquire()
        await asyncio.sleep(0.25)
        started = time.monotonic()
        await limiter.acquire()
        assert time.monotonic() - started < 0.1
