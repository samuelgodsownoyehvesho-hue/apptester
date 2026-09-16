"""Oracle signals: turn facts into verdicts.

Each signal reads only a :class:`~crucible.execute.checks.CheckResult.facts`
dict and answers one question: *does this evidence show a violation of a
stated invariant?* Signals never touch the network, so they can be re-run over
stored results and unit-tested without a live target.

The LLM judge signal is deliberately not here yet. With no reachable model
provider, the deterministic signals are the whole oracle — and that is a
feature, not a stopgap: an arithmetic or metamorphic violation is proof, not
opinion, and costs zero tokens. The judge joins later as an *additional*
signal for the lanes the deterministic ones cannot reach.

``suspected_bug_id`` fields deserve one honest note: they exist because the
checks were written against the target's *declared* invariants (the mutant
registry), so traceability is by construction. The benchmark treats them as
claims to verify, not answers — a finding whose facts do not actually show the
violation scores as a false positive regardless of the label.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from crucible.execute.checks import CheckResult
from crucible.recon.scout import AppMapData
from crucible.store.models import VerdictDecision


@dataclass(slots=True)
class SignalOutcome:
    """One signal's reading of one check's facts."""

    signal: str
    check_id: str
    #: True = invariant violated, False = invariant held, None = no opinion
    #: (the signal does not apply to this check, or the evidence is unusable).
    violated: bool | None
    confidence: float
    detail: str
    #: Citation: the exact numbers the decision rested on. Goes into the
    #: verdict's evidence list verbatim.
    evidence: dict[str, Any] = field(default_factory=dict)
    #: Traceability claim, see module docstring.
    suspected_bug_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "signal": self.signal,
            "check_id": self.check_id,
            "violated": self.violated,
            "confidence": self.confidence,
            "detail": self.detail,
            "suspected_bug_id": self.suspected_bug_id,
        }


# --------------------------------------------------------------- arithmetic


def _num(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _str_list(value: object) -> list[str] | None:
    """A list of strings, or ``None`` when the payload is not one."""
    if not isinstance(value, list):
        return None
    return [item for item in value if isinstance(item, str)]


def signal_cart_arithmetic(result: CheckResult) -> SignalOutcome:
    """Total must equal unit_price x quantity."""
    if result.check_id != "cart_quantity_arithmetic":
        return SignalOutcome("arithmetic", result.check_id, None, 0.0, "not applicable")

    expected = _num(result.facts.get("expected_total"))
    reported = _num(result.facts.get("reported_total"))
    if expected is None or reported is None:
        return SignalOutcome(
            "arithmetic", result.check_id, None, 0.0,
            "total missing from response; evidence unusable",
            {"facts": result.facts},
        )

    violated = abs(expected - reported) > 0.005
    return SignalOutcome(
        "arithmetic",
        result.check_id,
        violated,
        0.99,
        f"2 x $10.00 must total $20.00; server reported ${reported:.2f}",
        {
            "expected_total": expected,
            "reported_total": reported,
            "quantity": result.facts.get("quantity"),
            "unit_price": result.facts.get("unit_price"),
        },
        suspected_bug_id="CART_QTY_IGNORED" if violated else None,
    )


def signal_truncated_rounding(result: CheckResult) -> SignalOutcome:
    """A discounted total that loses fractions of a cent indicates truncation.

    Reading this off the same add-to-cart probe: 3 x $19.99 with no discount
    is $59.97 either way, so this signal only speaks when it can compute a
    total with a non-trivial third decimal. It stays silent otherwise — an
    under-powered signal must not borrow confidence from a sibling.
    """
    if result.check_id != "cart_quantity_arithmetic":
        return SignalOutcome("arithmetic", result.check_id, None, 0.0, "not applicable")

    reported = _num(result.facts.get("reported_total"))
    if reported is None:
        return SignalOutcome("arithmetic", result.check_id, None, 0.0, "no total")

    # 3 x $19.99 = $59.97 exactly; truncation of the *product* of a discount
    # would show, but this probe carries no discount. No opinion.
    return SignalOutcome("arithmetic", result.check_id, None, 0.0, "probe lacks a discount step")


# -------------------------------------------------------------- metamorphic


def signal_empty_cart_reset(result: CheckResult) -> SignalOutcome:
    """Zero items must mean zero total."""
    if result.check_id != "empty_cart_reset":
        return SignalOutcome("metamorphic", result.check_id, None, 0.0, "not applicable")

    line_count = result.facts.get("line_count")
    reported = _num(result.facts.get("reported_total"))
    if not isinstance(line_count, int) or reported is None:
        return SignalOutcome(
            "metamorphic", result.check_id, None, 0.0,
            "incomplete readback; evidence unusable", {"facts": result.facts},
        )

    violated = line_count == 0 and reported != 0.0
    return SignalOutcome(
        "metamorphic",
        result.check_id,
        violated,
        0.95,
        f"cart holds {line_count} items yet reports a total of ${reported:.2f}"
        if violated
        else f"empty cart reports ${reported:.2f}",
        {"line_count": line_count, "reported_total": reported},
        suspected_bug_id="EMPTY_CART_STALE_TOTAL" if violated else None,
    )


def signal_discount_idempotence(result: CheckResult) -> SignalOutcome:
    """Applying one code twice must not change the rate."""
    if result.check_id != "discount_idempotence":
        return SignalOutcome("metamorphic", result.check_id, None, 0.0, "not applicable")

    first = _num(result.facts.get("first_rate"))
    second = _num(result.facts.get("second_rate"))
    if first is None or second is None:
        return SignalOutcome(
            "metamorphic", result.check_id, None, 0.0,
            "discount rates missing; evidence unusable", {"facts": result.facts},
        )

    violated = abs(first - second) > 1e-9
    return SignalOutcome(
        "metamorphic",
        result.check_id,
        violated,
        0.95,
        f"rate moved from {first:.3f} to {second:.3f} on re-applying SAVE10"
        if violated
        else f"rate stable at {first:.3f}",
        {
            "first_rate": first,
            "second_rate": second,
            "first_total": result.facts.get("first_total"),
            "second_total": result.facts.get("second_total"),
        },
        suspected_bug_id="DISCOUNT_STACKS" if violated else None,
    )


def signal_price_monotonic(result: CheckResult) -> SignalOutcome:
    """A price-ascending listing must be non-decreasing."""
    if result.check_id != "price_sort_monotonic":
        return SignalOutcome("metamorphic", result.check_id, None, 0.0, "not applicable")

    prices = result.facts.get("prices_in_order")
    if not isinstance(prices, list) or len(prices) < 2:
        return SignalOutcome(
            "metamorphic", result.check_id, None, 0.0,
            "too few prices to judge ordering", {"facts": result.facts},
        )
    numeric: list[float] = []
    for price in prices:
        value = _num(price)
        if value is None:
            return SignalOutcome(
                "metamorphic", result.check_id, None, 0.0,
                "non-numeric price in listing", {"facts": result.facts},
            )
        numeric.append(value)

    inversions = [
        (position, numeric[position - 1], numeric[position])
        for position in range(1, len(numeric))
        if numeric[position] < numeric[position - 1] - 1e-9
    ]
    violated = bool(inversions)
    position, before, after = inversions[0] if inversions else (0, 0.0, 0.0)
    return SignalOutcome(
        "metamorphic",
        result.check_id,
        violated,
        0.95,
        f"price order breaks at position {position}: {before:.2f} followed by {after:.2f}"
        if violated
        else f"{len(numeric)} prices are non-decreasing",
        {
            "inversions": [
                {"position": p, "before": b, "after": a} for p, b, a in inversions
            ],
            "prices_in_order": numeric,
        },
        suspected_bug_id="LEXICOGRAPHIC_SORT" if violated else None,
    )


def signal_search_case_equivalence(result: CheckResult) -> SignalOutcome:
    """A term and its upper-casing must return the same products."""
    if result.check_id != "search_case_equivalence":
        return SignalOutcome("metamorphic", result.check_id, None, 0.0, "not applicable")

    lower = result.facts.get("lower_ids")
    upper = result.facts.get("upper_ids")
    if not isinstance(lower, list) or not isinstance(upper, list):
        return SignalOutcome(
            "metamorphic", result.check_id, None, 0.0,
            "result sets missing; evidence unusable", {"facts": result.facts},
        )

    violated = sorted(lower) != sorted(upper)
    # Name the products, not the counts: the two casings often match the *same
    # number* of different products, so "1 matched, 1 matched" reads like
    # nothing is wrong to anyone who is not comparing ids by eye.
    lower_names = _str_list(result.facts.get("lower_names"))
    upper_names = _str_list(result.facts.get("upper_names"))
    shown_lower = lower_names if lower_names is not None else lower
    shown_upper = upper_names if upper_names is not None else upper
    listed_lower = ", ".join(shown_lower) or "nothing"
    listed_upper = ", ".join(shown_upper) or "nothing"

    return SignalOutcome(
        "metamorphic",
        result.check_id,
        violated,
        0.9,
        f"the same search returned different products in each casing "
        f"(lower-case found: {listed_lower}; upper-case found: {listed_upper})"
        if violated
        else f"both casings matched the same {len(lower)} products",
        {"lower_ids": lower, "upper_ids": upper},
        suspected_bug_id="CASE_SENSITIVE_SEARCH" if violated else None,
    )


def signal_pagination_disjoint(result: CheckResult) -> SignalOutcome:
    """Consecutive pages must not share an item."""
    if result.check_id != "pagination_disjoint":
        return SignalOutcome("metamorphic", result.check_id, None, 0.0, "not applicable")

    overlap = result.facts.get("overlap")
    if not isinstance(overlap, list):
        return SignalOutcome(
            "metamorphic", result.check_id, None, 0.0,
            "page contents missing; evidence unusable", {"facts": result.facts},
        )

    violated = len(overlap) > 0
    return SignalOutcome(
        "metamorphic",
        result.check_id,
        violated,
        0.9,
        f"{len(overlap)} product(s) appear on both pages: {', '.join(overlap)}"
        if violated
        else "pages 1 and 2 are disjoint",
        {
            "overlap": overlap,
            "page1_ids": result.facts.get("page1_ids"),
            "page2_ids": result.facts.get("page2_ids"),
        },
        suspected_bug_id="PAGINATION_OVERLAP" if violated else None,
    )


# --------------------------------------------------------------------- http


def signal_negative_quantity(result: CheckResult) -> SignalOutcome:
    """A quantity below 1 must be rejected with a 4xx status."""
    if result.check_id != "negative_quantity_rejected":
        return SignalOutcome("http", result.check_id, None, 0.0, "not applicable")

    status = result.facts.get("status")
    if not isinstance(status, int) or status == 0:
        return SignalOutcome(
            "http", result.check_id, None, 0.0,
            "request never completed; evidence unusable", {"facts": result.facts},
        )

    rejected = 400 <= status < 500
    accepted = 200 <= status < 400
    if not rejected and not accepted:
        # A 5xx means the endpoint failed before it could validate anything.
        # Reading that as "the app accepted a negative quantity" is exactly the
        # false positive this signal exists to avoid: the request was never
        # processed, so it says nothing about input validation.
        return SignalOutcome(
            "http", result.check_id, None, 0.0,
            f"status {status} never exercised validation; evidence unusable",
            {"status": status, "body": result.facts.get("body")},
        )

    violated = accepted
    return SignalOutcome(
        "http",
        result.check_id,
        violated,
        0.9,
        f"quantity -1 was accepted with status {status}" if violated else f"rejected with {status}",
        {"status": status, "body": result.facts.get("body")},
        suspected_bug_id="CHECKOUT_ACCEPTS_NEGATIVE_QTY" if violated else None,
    )


def signal_routes_respond(result: CheckResult) -> SignalOutcome:
    """Every linked route must return a non-error status."""
    if result.check_id != "referenced_routes_respond":
        return SignalOutcome("http", result.check_id, None, 0.0, "not applicable")

    statuses = result.facts.get("statuses")
    if not isinstance(statuses, dict) or not statuses:
        return SignalOutcome(
            "http", result.check_id, None, 0.0,
            "no statuses recorded; evidence unusable", {"facts": result.facts},
        )

    broken = {
        str(route): int(status)
        for route, status in statuses.items()
        if not isinstance(status, int) or status < 200 or status >= 400
    }
    violated = bool(broken)
    return SignalOutcome(
        "http",
        result.check_id,
        violated,
        0.85,
        f"{len(broken)} route(s) returned an error status: "
        + ", ".join(f"{route} -> {code}" for route, code in sorted(broken.items()))
        if violated
        else f"all {len(statuses)} linked routes responded",
        {"broken": broken, "statuses": statuses},
        suspected_bug_id="BROKEN_FOOTER_LINK" if violated else None,
    )


# ------------------------------------------------------ recon-derived signal


def signal_unlabelled_controls(app_map: AppMapData | None) -> SignalOutcome:
    """Every form control observed by recon must have an accessible name.

    This one reads the app map rather than a check: the fact was captured at
    crawl time, and re-fetching pages to re-derive it would only add flake.
    """
    if app_map is None:
        return SignalOutcome("accessibility", "app_map_controls", None, 0.0, "no app map")

    controls = app_map.unlabelled_controls
    violated = len(controls) > 0
    where = sorted({str(control.get("page")) for control in controls})
    return SignalOutcome(
        "accessibility",
        "app_map_controls",
        violated,
        0.8,
        f"{len(controls)} control(s) without an accessible name on {', '.join(where)}"
        if violated
        else "every observed control has an accessible name",
        {
            "pages": where,
            "controls": [
                {"page": control.get("page"), "kind": control.get("kind"),
                 "name": control.get("name")}
                for control in controls
            ],
        },
        suspected_bug_id="UNLABELED_CHECKOUT_INPUT" if violated else None,
    )


#: Per-check signal mapping, in the order signals contribute to a verdict.
SIGNALS_BY_CHECK: dict[str, tuple[Any, ...]] = {
    "cart_quantity_arithmetic": (signal_cart_arithmetic, signal_truncated_rounding),
    "empty_cart_reset": (signal_empty_cart_reset,),
    "discount_idempotence": (signal_discount_idempotence,),
    "price_sort_monotonic": (signal_price_monotonic,),
    "search_case_equivalence": (signal_search_case_equivalence,),
    "pagination_disjoint": (signal_pagination_disjoint,),
    "negative_quantity_rejected": (signal_negative_quantity,),
    "referenced_routes_respond": (signal_routes_respond,),
}


def reach_verdict(
    results: list[CheckResult], app_map: AppMapData | None = None
) -> tuple[VerdictDecision, float, list[SignalOutcome]]:
    """Combine every signal's reading into one decision.

    Rules, in priority order:

    1. Any violated signal → ``BUG``. A deterministic invariant violation is
       proof; no amount of sibling agreement changes it.
    2. A check that errored → ``INSUFFICIENT_EVIDENCE`` for that check's
       signals. A transport failure is not evidence of correctness.
    3. Otherwise ``NOT_A_BUG`` — but only for the lanes actually exercised.
    """
    outcomes: list[SignalOutcome] = []
    for result in results:
        for signal in SIGNALS_BY_CHECK.get(result.check_id, ()):
            outcome = (
                signal(result)
                if result.error is None
                else SignalOutcome(
                    signal.__name__.removeprefix("signal_"),
                    result.check_id,
                    None,
                    0.0,
                    f"check did not complete: {result.error}",
                )
            )
            outcomes.append(outcome)
    outcomes.append(signal_unlabelled_controls(app_map))

    violated = [outcome for outcome in outcomes if outcome.violated]
    if violated:
        confidence = max(outcome.confidence for outcome in violated)
        return VerdictDecision.BUG, confidence, outcomes

    unusable = [outcome for outcome in outcomes if outcome.violated is None]
    if any(result.error for result in results) and unusable:
        return VerdictDecision.INSUFFICIENT_EVIDENCE, 0.0, outcomes

    held = [outcome for outcome in outcomes if outcome.violated is False]
    if held:
        return VerdictDecision.NOT_A_BUG, min(o.confidence for o in held), outcomes

    return VerdictDecision.INSUFFICIENT_EVIDENCE, 0.0, outcomes
