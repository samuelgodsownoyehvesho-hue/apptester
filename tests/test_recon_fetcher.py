"""Fetcher behaviour, focused on the egress guard.

The guard is enforced at the fetcher rather than only at the call site, so
these tests assert the property directly: an unauthorised host must raise
before any request is attempted. That is why they can run without a network —
the denial happens first.
"""

from __future__ import annotations

import pytest

from crucible.recon import EgressDenied, HttpFetcher, StaticFetcher


class TestEgressGuard:
    @pytest.mark.asyncio
    async def test_denies_a_host_outside_the_allowlist(self) -> None:
        fetcher = HttpFetcher(allowed_hosts=frozenset({"localhost"}))
        try:
            with pytest.raises(EgressDenied, match="egress allowlist"):
                await fetcher.fetch("http://evil.example.com/steal")
        finally:
            await fetcher.aclose()

    @pytest.mark.asyncio
    async def test_denies_a_disallowed_scheme(self) -> None:
        # file:// would read the local filesystem, and the allowlist is
        # host-based so it would not catch this on its own.
        fetcher = HttpFetcher()
        try:
            with pytest.raises(EgressDenied, match="unsupported scheme"):
                await fetcher.fetch("file:///etc/passwd")
        finally:
            await fetcher.aclose()

    @pytest.mark.asyncio
    async def test_allows_a_host_inside_the_allowlist(self) -> None:
        # Reaches the network layer, but a connection refused is a FetchResult
        # rather than an EgressDenied, which is the distinction under test.
        fetcher = HttpFetcher(allowed_hosts=frozenset({"127.0.0.1:9"}))
        try:
            result = await fetcher.fetch("http://127.0.0.1:9/")
            assert result.error is not None
            assert result.ok is False
        finally:
            await fetcher.aclose()

    def test_no_allowlist_permits_any_absolute_http_url(self) -> None:
        fetcher = HttpFetcher()
        # _check_egress is the guard; calling it directly avoids a request.
        fetcher._check_egress("https://anywhere.example.com/")


class TestFetchResult:
    def test_ok_for_success_statuses(self) -> None:
        assert StaticFetcher().pages == {}
        from crucible.recon import FetchResult

        assert FetchResult(url="u", status=200).ok is True
        assert FetchResult(url="u", status=302).ok is True

    def test_not_ok_for_error_statuses(self) -> None:
        from crucible.recon import FetchResult

        assert FetchResult(url="u", status=404).ok is False
        assert FetchResult(url="u", status=500).ok is False

    def test_not_ok_when_transport_failed(self) -> None:
        from crucible.recon import FetchResult

        result = FetchResult(url="u", status=200, error="ReadTimeout")
        assert result.ok is False

    def test_content_type_detection(self) -> None:
        from crucible.recon import FetchResult

        assert FetchResult(url="u", status=200, content_type="text/html; charset=utf-8").is_html
        assert FetchResult(url="u", status=200, content_type="application/json").is_json
        assert not FetchResult(url="u", status=200, content_type="image/png").is_html


class TestStaticFetcher:
    @pytest.mark.asyncio
    async def test_serves_configured_pages(self) -> None:
        fetcher = StaticFetcher(pages={"/x": (200, "<html>hi</html>")})
        result = await fetcher.fetch("http://host/x")
        assert result.status == 200
        assert result.text == "<html>hi</html>"

    @pytest.mark.asyncio
    async def test_returns_404_for_unknown_paths(self) -> None:
        result = await StaticFetcher().fetch("http://host/missing")
        assert result.status == 404

    @pytest.mark.asyncio
    async def test_matches_query_variants_separately_when_provided(self) -> None:
        fetcher = StaticFetcher(
            pages={"/p": (200, "base"), "/p?page=2": (200, "second")}
        )
        assert (await fetcher.fetch("http://host/p")).text == "base"
        assert (await fetcher.fetch("http://host/p?page=2")).text == "second"

    @pytest.mark.asyncio
    async def test_falls_back_to_the_path_when_query_is_unknown(self) -> None:
        fetcher = StaticFetcher(pages={"/p": (200, "base")})
        assert (await fetcher.fetch("http://host/p?page=9")).text == "base"

    @pytest.mark.asyncio
    async def test_infers_json_content_type(self) -> None:
        fetcher = StaticFetcher(pages={"/s.json": (200, "{}")})
        assert (await fetcher.fetch("http://host/s.json")).is_json

    @pytest.mark.asyncio
    async def test_records_calls(self) -> None:
        fetcher = StaticFetcher(pages={"/a": (200, "x")})
        await fetcher.fetch("http://host/a")
        await fetcher.fetch("http://host/b")
        assert fetcher.calls == ["http://host/a", "http://host/b"]
