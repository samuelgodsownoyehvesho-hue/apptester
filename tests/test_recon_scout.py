"""Scout crawling behaviour.

The bound tests matter most. A crawler without a page cap, a depth cap, and a
per-path variant cap can be driven into an unbounded crawl by a single faceted
search page, and the failure mode is a hung run rather than an error.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crucible.core.events import Event, EventBus, EventType
from crucible.recon import Scout, ScoutError, StaticFetcher

BASE = "http://localhost:3100"

HOME = """
<html><head><title>Home</title></head><body>
<h1>Nimbus Supply</h1>
<a href="/catalog">Catalog</a>
<a href="/cart">Cart</a>
<a href="/about">About</a>
<a href="https://example.com/external">External</a>
</body></html>
"""

CATALOG = """
<html><head><title>Catalog</title></head><body>
<h1>Catalog</h1>
<a href="/product/p-01">Laptop</a>
<a href="/catalog?page=2">Next</a>
<a href="/catalog?page=3">Last</a>
</body></html>
"""

CART = """
<html><head><title>Cart</title></head><body>
<h1>Cart</h1>
<form action="/api/cart" method="post">
  <label for="code">Discount code</label>
  <input id="code" name="code">
  <input name="mystery" placeholder="hint">
  <input type="submit" value="Apply">
</form>
</body></html>
"""


def _site(**extra: tuple[int, str]) -> StaticFetcher:
    pages = {
        "/": (200, HOME),
        "/catalog": (200, CATALOG),
        "/cart": (200, CART),
        "/about": (404, "<html><body>Not found</body></html>"),
    }
    pages.update(extra)
    return StaticFetcher(pages=pages)


class TestCrawling:
    @pytest.mark.asyncio
    async def test_crawls_the_entry_page(self) -> None:
        result = await Scout(BASE, _site()).crawl()
        assert result.routes[0].url == BASE
        assert result.routes[0].title == "Home"

    @pytest.mark.asyncio
    async def test_follows_same_origin_links(self) -> None:
        result = await Scout(BASE, _site()).crawl()
        assert {route.path for route in result.routes} >= {"/", "/catalog", "/cart"}

    @pytest.mark.asyncio
    async def test_does_not_crawl_off_origin(self) -> None:
        fetcher = _site()
        await Scout(BASE, fetcher).crawl()
        assert not any("example.com" in call for call in fetcher.calls)

    @pytest.mark.asyncio
    async def test_records_failed_routes(self) -> None:
        # A broken internal link is a fact the oracle may later judge.
        result = await Scout(BASE, _site()).crawl()
        assert "/about" in {route.path for route in result.failed_routes}

    @pytest.mark.asyncio
    async def test_records_transport_errors(self) -> None:
        fetcher = StaticFetcher(pages={"/": (200, HOME)})
        result = await Scout(BASE, fetcher, max_depth=1).crawl()
        # /catalog, /cart, /about are absent from the map, so they 404.
        assert all(route.status in {200, 404} for route in result.routes)

    @pytest.mark.asyncio
    async def test_deduplicates_visited_urls(self) -> None:
        fetcher = _site()
        await Scout(BASE, fetcher).crawl()
        assert len(fetcher.calls) == len(set(fetcher.calls))


class TestBounds:
    @pytest.mark.asyncio
    async def test_respects_max_pages(self) -> None:
        result = await Scout(BASE, _site(), max_pages=2).crawl()
        assert len(result.routes) == 2

    @pytest.mark.asyncio
    async def test_notes_that_the_map_is_partial_when_capped(self) -> None:
        result = await Scout(BASE, _site(), max_pages=1).crawl()
        assert any("partial" in note for note in result.notes)

    @pytest.mark.asyncio
    async def test_respects_max_depth(self) -> None:
        # At depth 0 only the entry page is fetched; its links are recorded
        # but never followed.
        fetcher = _site()
        result = await Scout(BASE, fetcher, max_depth=0).crawl()
        assert len(result.routes) == 1

        # robots.txt is fetched before crawling starts, so it is expected in
        # the call log and must be excluded before counting page fetches.
        page_calls = [call for call in fetcher.calls if not call.endswith("/robots.txt")]
        assert page_calls == [BASE]

    @pytest.mark.asyncio
    async def test_caps_query_variants_per_path(self) -> None:
        # /catalog?page=N would otherwise expand without limit, since the
        # static fetcher serves the same page for every variant.
        fetcher = _site()
        await Scout(BASE, fetcher, max_variants_per_path=1).crawl()
        catalog_calls = [call for call in fetcher.calls if call.startswith(f"{BASE}/catalog")]
        assert len(catalog_calls) == 1


class TestRobots:
    @pytest.mark.asyncio
    async def test_skips_disallowed_paths(self) -> None:
        fetcher = _site(**{"/robots.txt": (200, "User-agent: *\nDisallow: /cart\n")})
        result = await Scout(BASE, fetcher).crawl()
        assert "/cart" not in {route.path for route in result.routes}
        assert "/catalog" in {route.path for route in result.routes}

    @pytest.mark.asyncio
    async def test_records_that_paths_were_excluded(self) -> None:
        fetcher = _site(**{"/robots.txt": (200, "User-agent: *\nDisallow: /cart\n")})
        result = await Scout(BASE, fetcher).crawl()
        assert any("robots.txt" in note for note in result.notes)

    @pytest.mark.asyncio
    async def test_refuses_to_start_when_the_entry_point_is_disallowed(self) -> None:
        fetcher = _site(**{"/robots.txt": (200, "User-agent: *\nDisallow: /\n")})
        with pytest.raises(ScoutError, match="disallows the entry point"):
            await Scout(BASE, fetcher).crawl()

    @pytest.mark.asyncio
    async def test_missing_robots_permits_everything(self) -> None:
        result = await Scout(BASE, _site()).crawl()
        assert len(result.routes) > 1

    @pytest.mark.asyncio
    async def test_ignores_rules_for_other_user_agents(self) -> None:
        robots = "User-agent: SomeOtherBot\nDisallow: /\n"
        fetcher = _site(**{"/robots.txt": (200, robots)})
        result = await Scout(BASE, fetcher).crawl()
        assert len(result.routes) > 1

    @pytest.mark.asyncio
    async def test_can_be_disabled(self) -> None:
        fetcher = _site(**{"/robots.txt": (200, "User-agent: *\nDisallow: /\n")})
        result = await Scout(BASE, fetcher, respect_robots=False).crawl()
        assert len(result.routes) >= 1


class TestFactRecording:
    @pytest.mark.asyncio
    async def test_records_forms(self) -> None:
        result = await Scout(BASE, _site()).crawl()
        actions = {form["action"] for form in result.forms}
        assert f"{BASE}/api/cart" in actions

    @pytest.mark.asyncio
    async def test_records_unlabelled_controls(self) -> None:
        result = await Scout(BASE, _site()).crawl()
        names = {control["name"] for control in result.unlabelled_controls}
        # The labelled "code" field must not appear; the placeholder-only
        # "mystery" field must.
        assert "mystery" in names
        assert "code" not in names

    @pytest.mark.asyncio
    async def test_route_counts_labelled_and_unlabelled_controls(self) -> None:
        result = await Scout(BASE, _site()).crawl()
        cart = next(route for route in result.routes if route.path == "/cart")
        assert cart.form_count == 1
        assert cart.unlabelled_control_count == 1

    @pytest.mark.asyncio
    async def test_route_paths_are_unique_and_sorted(self) -> None:
        result = await Scout(BASE, _site()).crawl()
        paths = result.route_paths
        assert paths == sorted(set(paths))

    @pytest.mark.asyncio
    async def test_serialises_to_dict(self) -> None:
        payload = (await Scout(BASE, _site()).crawl()).as_dict()
        assert payload["base_url"] == BASE
        assert payload["route_count"] == len(payload["routes"])  # type: ignore[arg-type]


class TestEvents:
    @pytest.mark.asyncio
    async def test_emits_start_and_finish(self) -> None:
        seen: list[Event] = []
        bus = EventBus("run_1")
        bus.subscribe(seen.append)

        await Scout(BASE, _site(), bus=bus).crawl()

        types = [event.type for event in seen]
        assert EventType.RECON_STARTED in types
        assert EventType.RECON_FINISHED in types

    @pytest.mark.asyncio
    async def test_emits_one_event_per_page(self) -> None:
        seen: list[Event] = []
        bus = EventBus("run_1")
        bus.subscribe(seen.append)

        result = await Scout(BASE, _site(), bus=bus).crawl()

        visits = [e for e in seen if e.type is EventType.RECON_PAGE_VISITED]
        assert len(visits) == len(result.routes)
        assert all("url" in event.payload for event in visits)


class TestOpenApiImport:
    @pytest.mark.asyncio
    async def test_imports_operations_from_a_spec(self, tmp_path: Path) -> None:
        spec = {
            "openapi": "3.0.0",
            "paths": {
                "/api/products": {"get": {"summary": "List products"}},
                "/api/cart": {"post": {"summary": "Add to cart"}},
            },
        }
        fetcher = _site(**{"/openapi.json": (200, json.dumps(spec))})
        result = await Scout(BASE, fetcher).crawl()

        await Scout(BASE, fetcher).load_openapi(f"{BASE}/openapi.json", result)

        operations = {(op["method"], op["path"]) for op in result.api_operations}
        assert ("GET", "/api/products") in operations
        assert ("POST", "/api/cart") in operations

    @pytest.mark.asyncio
    async def test_records_unavailable_spec(self) -> None:
        result = await Scout(BASE, _site()).crawl()
        await Scout(BASE, _site()).load_openapi(f"{BASE}/missing.json", result)
        assert any("unavailable" in note for note in result.notes)

    @pytest.mark.asyncio
    async def test_records_invalid_json(self) -> None:
        fetcher = _site(**{"/bad.json": (200, "not json{")})
        result = await Scout(BASE, fetcher).crawl()
        await Scout(BASE, fetcher).load_openapi(f"{BASE}/bad.json", result)
        assert any("not valid JSON" in note for note in result.notes)

    @pytest.mark.asyncio
    async def test_reports_import_summary_once(self) -> None:
        spec = {"openapi": "3.0.0", "paths": {"/a": {"get": {}}, "/b": {"get": {}}}}
        fetcher = _site(**{"/openapi.json": (200, json.dumps(spec))})
        result = await Scout(BASE, fetcher).crawl()
        await Scout(BASE, fetcher).load_openapi(f"{BASE}/openapi.json", result)
        summaries = [note for note in result.notes if "Imported" in note]
        assert len(summaries) == 1


class TestConstruction:
    def test_rejects_a_relative_base_url(self) -> None:
        with pytest.raises(ScoutError, match="absolute"):
            Scout("/no-scheme", _site())

    def test_rejects_a_non_http_scheme(self) -> None:
        with pytest.raises(ScoutError, match="absolute"):
            Scout("file:///etc/passwd", _site())
