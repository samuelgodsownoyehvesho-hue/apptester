"""Tests for the deterministic oracle signals.

Each signal is exercised twice: once with facts showing the invariant held,
and once with facts showing a violation, taken from the behavioural twin of
the guinea pig app. Silence (``None``) is asserted where a signal must have no
opinion — an under-powered signal borrowing confidence from a sibling is the
failure mode these tests exist to prevent.
"""

from __future__ import annotations

import pytest

from crucible.execute.checks import CheckResult
from crucible.oracle.signals import (
    SIGNALS_BY_CHECK,
    SignalOutcome,
    reach_verdict,
    signal_unlabelled_controls,
)
from crucible.recon.scout import AppMapData
from crucible.store.models import VerdictDecision


def _run(check_id: str, facts: dict[str, object]) -> SignalOutcome:
    result = CheckResult(check_id=check_id, lane="test", observation="", facts=facts)
    signal = SIGNALS_BY_CHECK[check_id][0]
    return signal(result)


class TestCartArithmetic:
    def test_holds_when_total_matches(self) -> None:
        outcome = _run(
            "cart_quantity_arithmetic",
            {"expected_total": 20.0, "reported_total": 20.0, "quantity": 2, "unit_price": 10.0},
        )
        assert outcome.violated is False
        assert outcome.suspected_bug_id is None

    def test_violated_when_quantity_ignored(self) -> None:
        outcome = _run(
            "cart_quantity_arithmetic",
            {"expected_total": 20.0, "reported_total": 10.0, "quantity": 2, "unit_price": 10.0},
        )
        assert outcome.violated is True
        assert outcome.suspected_bug_id == "CART_QTY_IGNORED"

    def test_no_opinion_when_total_missing(self) -> None:
        outcome = _run("cart_quantity_arithmetic", {"expected_total": 20.0})
        assert outcome.violated is None
        assert outcome.confidence == 0.0

    def test_rounding_is_not_read_off_the_quantity_probe(self) -> None:
        """Rounding has its own probe, because this one has no discount.

        The quantity probe adds 2 x $10.00, which is exact either way. A
        rounding signal attached here could only ever stay silent, which is how
        the truncation defect went undetected while looking covered.
        """
        assert len(SIGNALS_BY_CHECK["cart_quantity_arithmetic"]) == 1
        assert len(SIGNALS_BY_CHECK["rounding_precision"]) == 1


class TestRoundingPrecision:
    def test_holds_when_rounded(self) -> None:
        # $1.07 less 20% is $0.85600, which rounds to $0.86.
        outcome = _run(
            "rounding_precision",
            {"subtotal": 1.07, "discount_rate": 0.2, "reported_total": 0.86},
        )
        assert outcome.violated is False
        assert outcome.suspected_bug_id is None

    def test_violated_when_truncated(self) -> None:
        outcome = _run(
            "rounding_precision",
            {"subtotal": 1.07, "discount_rate": 0.2, "reported_total": 0.85},
        )
        assert outcome.violated is True
        assert outcome.suspected_bug_id == "TRUNCATED_ROUNDING"

    def test_no_opinion_when_the_two_agree(self) -> None:
        # No discount: 1.07 rounds and truncates identically, so the probe has
        # nothing to say and must not claim the invariant held.
        outcome = _run(
            "rounding_precision",
            {"subtotal": 1.07, "discount_rate": 0.0, "reported_total": 1.07},
        )
        assert outcome.violated is None
        assert outcome.confidence == 0.0

    def test_no_opinion_when_money_fields_are_missing(self) -> None:
        outcome = _run("rounding_precision", {"subtotal": 1.07})
        assert outcome.violated is None


class TestEmptyCartReset:
    def test_holds(self) -> None:
        outcome = _run("empty_cart_reset", {"line_count": 0, "reported_total": 0.0})
        assert outcome.violated is False

    def test_violated_stale_total(self) -> None:
        outcome = _run("empty_cart_reset", {"line_count": 0, "reported_total": 10.0})
        assert outcome.violated is True
        assert outcome.suspected_bug_id == "EMPTY_CART_STALE_TOTAL"

    def test_items_present_is_not_violation(self) -> None:
        # Lines remaining means the removal itself failed; the invariant is
        # about the zero-item case only.
        outcome = _run("empty_cart_reset", {"line_count": 2, "reported_total": 30.0})
        assert outcome.violated is False


class TestDiscountIdempotence:
    def test_holds(self) -> None:
        outcome = _run(
            "discount_idempotence", {"first_rate": 0.1, "second_rate": 0.1}
        )
        assert outcome.violated is False

    def test_violated_stacking(self) -> None:
        outcome = _run(
            "discount_idempotence", {"first_rate": 0.1, "second_rate": 0.19}
        )
        assert outcome.violated is True
        assert outcome.suspected_bug_id == "DISCOUNT_STACKS"


class TestPriceMonotonic:
    def test_holds(self) -> None:
        outcome = _run(
            "price_sort_monotonic", {"prices_in_order": [9.99, 19.99, 149.99]}
        )
        assert outcome.violated is False

    def test_violated_lexicographic(self) -> None:
        outcome = _run(
            "price_sort_monotonic", {"prices_in_order": [149.99, 19.99, 9.99]}
        )
        assert outcome.violated is True
        assert outcome.suspected_bug_id == "LEXICOGRAPHIC_SORT"
        assert outcome.evidence["inversions"][0]["before"] == pytest.approx(149.99)

    def test_too_few_prices_is_no_opinion(self) -> None:
        outcome = _run("price_sort_monotonic", {"prices_in_order": [9.99]})
        assert outcome.violated is None


class TestSearchEquivalence:
    def test_holds(self) -> None:
        outcome = _run(
            "search_case_equivalence",
            {"lower_ids": ["p-01", "p-02"], "upper_ids": ["p-02", "p-01"]},
        )
        assert outcome.violated is False

    def test_violated(self) -> None:
        outcome = _run(
            "search_case_equivalence", {"lower_ids": [], "upper_ids": ["p-01"]}
        )
        assert outcome.violated is True
        assert outcome.suspected_bug_id == "CASE_SENSITIVE_SEARCH"


class TestPagination:
    def test_holds(self) -> None:
        outcome = _run(
            "pagination_disjoint",
            {"overlap": [], "page1_ids": ["p-01"], "page2_ids": ["p-02"]},
        )
        assert outcome.violated is False

    def test_violated_overlap(self) -> None:
        outcome = _run(
            "pagination_disjoint",
            {"overlap": ["p-06"], "page1_ids": ["p-06"], "page2_ids": ["p-06"]},
        )
        assert outcome.violated is True
        assert outcome.suspected_bug_id == "PAGINATION_OVERLAP"


class TestNegativeQuantity:
    def test_holds_on_400(self) -> None:
        outcome = _run("negative_quantity_rejected", {"status": 400, "body": None})
        assert outcome.violated is False

    def test_violated_on_201(self) -> None:
        outcome = _run("negative_quantity_rejected", {"status": 201, "body": None})
        assert outcome.violated is True
        assert outcome.suspected_bug_id == "CHECKOUT_ACCEPTS_NEGATIVE_QTY"

    def test_transport_failure_is_no_opinion(self) -> None:
        outcome = _run("negative_quantity_rejected", {"status": 0, "body": None})
        assert outcome.violated is None

    def test_server_error_is_no_opinion(self) -> None:
        # A 501 means the endpoint never ran, so it is not evidence that a
        # negative quantity was accepted. Treating 5xx as acceptance is a
        # false positive against any target that lacks the cart API.
        outcome = _run("negative_quantity_rejected", {"status": 501, "body": None})
        assert outcome.violated is None
        assert outcome.confidence == 0.0


class TestRoutesRespond:
    def test_holds(self) -> None:
        outcome = _run(
            "referenced_routes_respond",
            {"statuses": {"/": 200, "/catalog": 200, "/returns": 200}},
        )
        assert outcome.violated is False

    def test_violated_broken_link(self) -> None:
        outcome = _run(
            "referenced_routes_respond",
            {"statuses": {"/": 200, "/returns": 404}},
        )
        assert outcome.violated is True
        assert outcome.suspected_bug_id == "BROKEN_FOOTER_LINK"

    def test_5xx_counts_as_broken(self) -> None:
        outcome = _run(
            "referenced_routes_respond", {"statuses": {"/": 200, "/cart": 500}}
        )
        assert outcome.violated is True


class TestUnlabelledControls:
    def test_no_app_map_is_no_opinion(self) -> None:
        outcome = signal_unlabelled_controls(None)
        assert outcome.violated is None

    def test_clean_map_holds(self) -> None:
        app_map = AppMapData(base_url="http://test")
        outcome = signal_unlabelled_controls(app_map)
        assert outcome.violated is False

    def test_violated(self) -> None:
        app_map = AppMapData(base_url="http://test")
        app_map.unlabelled_controls.append(
            {"page": "/checkout", "kind": "input", "name": "email"}
        )
        outcome = signal_unlabelled_controls(app_map)
        assert outcome.violated is True
        assert outcome.suspected_bug_id == "UNLABELED_CHECKOUT_INPUT"


class TestReachVerdict:
    def test_any_violation_is_bug(self) -> None:
        results = [
            CheckResult(
                check_id="cart_quantity_arithmetic",
                lane="arithmetic",
                observation="",
                facts={"expected_total": 20.0, "reported_total": 10.0},
            )
        ]
        decision, confidence, outcomes = reach_verdict(results)
        assert decision is VerdictDecision.BUG
        assert confidence == pytest.approx(0.99)
        assert any(outcome.violated is True for outcome in outcomes)

    def test_all_held_is_not_a_bug(self) -> None:
        results = [
            CheckResult(
                check_id="cart_quantity_arithmetic",
                lane="arithmetic",
                observation="",
                facts={"expected_total": 20.0, "reported_total": 20.0},
            )
        ]
        decision, _, outcomes = reach_verdict(results)
        assert decision is VerdictDecision.NOT_A_BUG
        assert all(outcome.violated is not True for outcome in outcomes)

    def test_errored_check_is_insufficient_evidence(self) -> None:
        results = [
            CheckResult(
                check_id="cart_quantity_arithmetic",
                lane="arithmetic",
                observation="",
                facts={},
                error="ConnectionError: boom",
            )
        ]
        decision, confidence, outcomes = reach_verdict(results)
        assert decision is VerdictDecision.INSUFFICIENT_EVIDENCE
        assert confidence == 0.0
        assert outcomes[0].violated is None

    def test_empty_results_is_insufficient(self) -> None:
        decision, confidence, _ = reach_verdict([])
        assert decision is VerdictDecision.INSUFFICIENT_EVIDENCE
        assert confidence == 0.0
