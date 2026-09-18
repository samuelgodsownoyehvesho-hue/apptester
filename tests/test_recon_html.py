"""HTML fact extraction.

The accessible-name assertions mirror the real defect seeded in the guinea pig
checkout page: a `<p>` label above an input plus a placeholder, with no actual
association. A parser that counted the placeholder or the adjacent text would
report that control as labelled and the defect would become undetectable.
"""

from __future__ import annotations

from crucible.execute.checks import CheckResult
from crucible.oracle.signals import SIGNALS_BY_CHECK, signal_ui_interaction
from crucible.recon.html import extract_page_facts

BASE = "http://localhost:3100"


def _fields(html: str) -> list[object]:
    facts = extract_page_facts(html, BASE)
    return [f for form in facts.forms for f in form.fields] + list(facts.loose_fields)


class TestLinks:
    def test_collects_same_origin_links(self) -> None:
        facts = extract_page_facts('<a href="/catalog">C</a><a href="/cart">K</a>', BASE)
        assert facts.links == [f"{BASE}/catalog", f"{BASE}/cart"]

    def test_resolves_relative_links(self) -> None:
        # urljoin treats the final segment of the base as a file, so a relative
        # href resolves against the parent directory. This matches browser
        # behaviour and is why the real app uses absolute hrefs.
        facts = extract_page_facts('<a href="product/p-01">P</a>', f"{BASE}/catalog")
        assert facts.links == [f"{BASE}/product/p-01"]

    def test_relative_link_from_a_directory_base(self) -> None:
        facts = extract_page_facts('<a href="p-01">P</a>', f"{BASE}/catalog/")
        assert facts.links == [f"{BASE}/catalog/p-01"]

    def test_separates_external_links(self) -> None:
        facts = extract_page_facts('<a href="https://example.com/x">E</a>', BASE)
        assert facts.links == []
        assert facts.external_links == ["https://example.com/x"]

    def test_ignores_non_navigational_hrefs(self) -> None:
        facts = extract_page_facts(
            '<a href="#top">a</a><a href="mailto:a@b.c">b</a><a href="javascript:void(0)">c</a>',
            BASE,
        )
        assert facts.links == []

    def test_strips_fragments(self) -> None:
        facts = extract_page_facts('<a href="/catalog#section">C</a>', BASE)
        assert facts.links == [f"{BASE}/catalog"]

    def test_deduplicates_links(self) -> None:
        facts = extract_page_facts('<a href="/a">1</a><a href="/a">2</a>', BASE)
        assert facts.links == [f"{BASE}/a"]


class TestForms:
    def test_captures_form_action_and_method(self) -> None:
        facts = extract_page_facts('<form action="/api/cart" method="post"></form>', BASE)
        assert len(facts.forms) == 1
        assert facts.forms[0].action == f"{BASE}/api/cart"
        assert facts.forms[0].method == "post"

    def test_defaults_action_to_the_page_and_method_to_get(self) -> None:
        facts = extract_page_facts("<form></form>", BASE)
        assert facts.forms[0].action == BASE
        assert facts.forms[0].method == "get"

    def test_captures_control_attributes(self) -> None:
        html = '<form><input type="email" name="email" id="email" required></form>'
        field = _fields(html)[0]
        assert field.kind == "email"  # type: ignore[attr-defined]
        assert field.name == "email"  # type: ignore[attr-defined]
        assert field.required is True  # type: ignore[attr-defined]

    def test_select_and_textarea_are_controls(self) -> None:
        html = "<form><select name='s'></select><textarea name='t'></textarea></form>"
        kinds = {f.kind for f in _fields(html)}  # type: ignore[attr-defined]
        assert kinds == {"select", "textarea"}


class TestAccessibleNames:
    def test_label_for_is_recognised(self) -> None:
        html = '<label for="email">Email</label><input type="email" id="email">'
        field = _fields(html)[0]
        assert field.has_label is True  # type: ignore[attr-defined]
        assert field.label_text == "Email"  # type: ignore[attr-defined]

    def test_label_appearing_after_the_control_is_recognised(self) -> None:
        # Labels are frequently emitted after the input; a single-pass parser
        # that resolved immediately would miss this.
        html = '<input type="email" id="email"><label for="email">Email</label>'
        field = _fields(html)[0]
        assert field.has_label is True  # type: ignore[attr-defined]
        assert field.label_text == "Email"  # type: ignore[attr-defined]

    def test_wrapped_control_is_recognised(self) -> None:
        html = '<label>Email <input type="email" name="email"></label>'
        assert _fields(html)[0].has_label is True  # type: ignore[attr-defined]

    def test_aria_label_is_recognised(self) -> None:
        html = '<input type="email" name="email" aria-label="Email address">'
        assert _fields(html)[0].has_label is True  # type: ignore[attr-defined]

    def test_placeholder_is_not_an_accessible_name(self) -> None:
        # This is the seeded defect: a placeholder is a hint, not a label.
        html = '<input type="email" name="email" placeholder="you@example.com">'
        field = _fields(html)[0]
        assert field.has_label is False  # type: ignore[attr-defined]
        assert field.placeholder == "you@example.com"  # type: ignore[attr-defined]

    def test_adjacent_paragraph_text_is_not_an_accessible_name(self) -> None:
        # Also the seeded defect: visually a label, structurally nothing.
        html = '<p>Email address</p><input type="email" name="email" placeholder="you@example.com">'
        assert _fields(html)[0].has_label is False  # type: ignore[attr-defined]

    def test_id_without_a_matching_label_is_not_labelled(self) -> None:
        html = '<label for="other">Email</label><input type="email" id="email">'
        assert _fields(html)[0].has_label is False  # type: ignore[attr-defined]

    def test_reports_unlabelled_controls(self) -> None:
        html = (
            '<form><label for="ok">Fine</label><input id="ok" name="ok">'
            '<input name="bad" placeholder="hint"></form>'
        )
        facts = extract_page_facts(html, BASE)
        unlabelled = facts.unlabelled_controls
        assert [f.name for f in unlabelled] == ["bad"]

    def test_hidden_and_submit_controls_are_excluded_from_unlabelled(self) -> None:
        # A submit button's value is normally its name and a hidden field is not
        # user-facing, so neither is an accessibility defect.
        html = '<form><input type="hidden" name="csrf"><input type="submit" value="Go"></form>'
        facts = extract_page_facts(html, BASE)
        assert facts.unlabelled_controls == []


class TestPageMetadata:
    def test_extracts_title(self) -> None:
        facts = extract_page_facts("<title>Catalog</title>", BASE)
        assert facts.title == "Catalog"

    def test_extracts_headings(self) -> None:
        facts = extract_page_facts("<h1>Hello</h1><h2>World</h2>", BASE)
        assert facts.headings == ["Hello", "World"]

    def test_title_is_none_when_absent(self) -> None:
        assert extract_page_facts("<p>x</p>", BASE).title is None

    def test_handles_malformed_html_without_raising(self) -> None:
        # Real pages are frequently invalid; a parse failure must not abort a
        # crawl that is otherwise gathering useful data.
        facts = extract_page_facts("<div><a href='/x'>unclosed", BASE)
        assert f"{BASE}/x" in facts.links

    def test_serialises_to_dict(self) -> None:
        payload = extract_page_facts("<title>T</title>", BASE).as_dict()
        assert payload["url"] == BASE
        assert payload["title"] == "T"


def _elements(html: str) -> list[object]:
    return list(extract_page_facts(html, BASE).elements)


class TestElementInventory:
    """The interaction lane has nothing to press without this inventory.

    Every control a page offers has to be named here, because the lane works
    only from what reconnaissance found: a button this misses is a button that
    is never tested, and the run still reports as complete.
    """

    def test_buttons_and_links_are_inventoried_with_their_text(self) -> None:
        html = """
        <html><body>
          <button>Add to cart</button>
          <button aria-label="Close dialog"></button>
          <a href="/catalog">Browse the catalog</a>
        </body></html>
        """
        elements = _elements(html)
        assert {(e.kind, e.label) for e in elements} == {
            ("button", "Add to cart"),
            ("button", "Close dialog"),
            ("link", "Browse the catalog"),
        }

    def test_an_id_gives_a_selector_a_static_parse_can_trust(self) -> None:
        elements = _elements('<html><body><button id="buy-now">Buy</button></body></html>')
        assert elements[0].selector == "#buy-now"

    def test_hidden_inputs_have_nothing_to_press(self) -> None:
        """A hidden field is not a control, so pressing it is not a test."""
        elements = _elements(
            '<html><body><input type="hidden" name="csrf" value="x"></body></html>'
        )
        assert elements == []

    def test_select_options_are_recorded_for_exercise(self) -> None:
        elements = _elements(
            '<html><body><select name="sort">'
            "<option value='a'>Price: low to high</option>"
            "<option value='b'>Price: high to low</option>"
            "</select></body></html>"
        )
        assert elements[0].kind == "select"
        assert elements[0].options == ("Price: low to high", "Price: high to low")

    def test_a_click_handler_makes_a_non_semantic_element_a_control(self) -> None:
        elements = _elements('<html><body><div onclick="open()">Open menu</div></body></html>')
        assert elements[0].kind == "clickable"
        assert elements[0].label == "Open menu"

    def test_every_control_on_a_page_is_kept(self) -> None:
        """A low ceiling would read as a clean page rather than a truncated one."""
        buttons = "".join(f"<button>Action {index}</button>" for index in range(120))
        elements = _elements(f"<html><body>{buttons}</body></html>")
        assert len(elements) == 120


class TestInteractionSignal:
    """A control that breaks must reach the report as a defect."""

    def test_a_javascript_error_is_a_violation(self) -> None:
        outcome = signal_ui_interaction(
            CheckResult(
                check_id="ui_interaction",
                lane="ui",
                observation="",
                facts={
                    "route": "/checkout",
                    "kind": "button",
                    "label": "Place order",
                    "failure": "js_error",
                    "js_error": "TypeError: cannot read property 'total' of null",
                },
            )
        )
        assert outcome.violated is True
        assert "Place order" in outcome.detail
        assert "JavaScript error" in outcome.detail
        # A broken button is not one of the target's declared defects, so no id
        # is claimed: inventing one would corrupt the benchmark with a fake hit.
        assert outcome.suspected_bug_id is None

    def test_a_control_that_behaved_is_not_a_violation(self) -> None:
        outcome = signal_ui_interaction(
            CheckResult(
                check_id="ui_interaction",
                lane="ui",
                observation="",
                facts={"route": "/", "kind": "link", "label": "Catalog", "failure": None},
            )
        )
        assert outcome.violated is False

    def test_a_hanging_control_is_a_violation(self) -> None:
        outcome = signal_ui_interaction(
            CheckResult(
                check_id="ui_interaction",
                lane="ui",
                observation="",
                facts={
                    "route": "/",
                    "kind": "button",
                    "label": "Search",
                    "failure": "timeout",
                    "detail": "TimeoutError: 15000ms exceeded",
                },
            )
        )
        assert outcome.violated is True
        assert "never finished" in outcome.detail

    def test_every_interaction_check_has_a_signal_registered(self) -> None:
        assert "ui_interaction" in SIGNALS_BY_CHECK
