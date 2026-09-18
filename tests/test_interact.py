"""The interaction lane's reporting contract, without launching a browser.

What matters here is not the clicking itself but what the lane hands to the rest
of the system: a broken control has to arrive as evidence the oracle can judge,
and a control that could not be judged has to arrive as nothing at all.
"""

from __future__ import annotations

import pytest

from crucible.execute.interact import (
    INTERACTION_CHECK_ID,
    InteractionReport,
    InteractionResult,
    exercise_elements,
)
from crucible.pipeline import _elements_by_route
from crucible.recon.scout import AppMapData


def _broken(**overrides: object) -> InteractionResult:
    values: dict[str, object] = {
        "route": "/checkout",
        "kind": "button",
        "label": "Place order",
        "selector": "#place-order",
        "failure": "js_error",
        "js_error": "TypeError: total is null",
    }
    values.update(overrides)
    return InteractionResult(**values)  # type: ignore[arg-type]


class TestEvidenceHandoff:
    """A broken control must reach the report as a defect, not as a side note."""

    def test_a_failure_becomes_an_oracle_check_result(self) -> None:
        evidence = _broken().as_check_result()
        assert evidence.check_id == INTERACTION_CHECK_ID
        assert evidence.lane == "ui"
        assert evidence.facts["failure"] == "js_error"
        assert evidence.facts["js_error"] == "TypeError: total is null"
        assert "Place order" in evidence.observation

    def test_the_control_is_named_by_its_label_when_it_has_one(self) -> None:
        assert _broken(label="Place order").name == "Place order"
        # A control with no visible text still has to be nameable, so the
        # selector stands in rather than reporting an anonymous failure.
        assert _broken(label="", selector="#place-order").name == "#place-order"

    def test_a_healthy_control_reports_no_failure(self) -> None:
        result = InteractionResult(route="/", kind="link", label="Catalog", selector="a")
        assert result.ok is True
        assert result.as_check_result().facts["failure"] is None

    def test_a_skipped_control_is_not_a_failure(self) -> None:
        """An unreachable page says nothing about the controls it contains."""
        result = _broken(skipped=True, detail="the page did not load")
        report = InteractionReport(results=[result])
        assert report.failures == []

    def test_the_report_counts_what_it_found(self) -> None:
        report = InteractionReport(results=[_broken(), _broken(label="Save", failure=None)])
        assert report.as_dict()["elements"] == 2
        assert report.as_dict()["failures"] == 1


class TestElementGrouping:
    """The lane works per page, so the inventory has to be grouped by route."""

    def test_elements_are_grouped_by_the_page_they_live_on(self) -> None:
        app_map = AppMapData(base_url="http://shop.test")
        app_map.elements = [
            {"page": "/", "kind": "link", "label": "Catalog", "selector": "a"},
            {"page": "/cart", "kind": "button", "label": "Checkout", "selector": "#go"},
            {"page": "/", "kind": "button", "label": "Search", "selector": "#s"},
        ]

        grouped = _elements_by_route(app_map)

        assert sorted(grouped) == ["/", "/cart"]
        assert [element["label"] for element in grouped["/"]] == ["Catalog", "Search"]
        # The page tag is stripped: the lane already knows which page it is on.
        assert "page" not in grouped["/cart"][0]

    def test_an_entry_without_a_page_is_ignored(self) -> None:
        app_map = AppMapData(base_url="http://shop.test")
        app_map.elements = [{"kind": "button", "label": "orphan"}]
        assert _elements_by_route(app_map) == {}


class TestNoElements:
    """An empty inventory is a reported reason, never a silent clean pass."""

    @pytest.mark.asyncio
    async def test_no_elements_returns_a_note_without_a_browser(self, tmp_path) -> None:
        from crucible.store.artifacts import ArtifactStore

        report = await exercise_elements(
            "http://shop.test", {"/": []}, "run_1", ArtifactStore(tmp_path / "a")
        )

        assert report.results == []
        assert report.note is not None
        assert "no interactive elements" in report.note
