"""Turn violated verdicts into deduplicated findings.

Triage consumes :class:`~crucible.oracle.signals.SignalOutcome` lists and
emits :class:`~crucible.store.models.Finding` rows. The rules are narrow on
purpose:

* Only signals that actually *violated* an invariant can contribute to a
  finding. "Not applicable" and "check errored" are recorded facts, not
  defects.
* Clustering is per signal **and** per suspected bug id, so one defect
  evidenced by several probes collapses to one finding instead of five.
* Matched bug ids ride along as claims for the benchmark to verify. They are
  never treated as truth here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from crucible.core.events import EventBus, EventType
from crucible.oracle.signals import SignalOutcome
from crucible.store.models import (
    Finding,
    FindingStatus,
    Severity,
    VerdictDecision,
)


@dataclass(frozen=True, slots=True)
class FindingDraft:
    """A finding before persistence, for tests and callers that only report."""

    key: str
    title: str
    severity: Severity
    root_cause: str
    evidence: dict[str, Any]
    suspected_bug_id: str | None
    #: What to change to fix it, when the check has a known remedy.
    suggested_fix: str | None = None


#: Signal -> severity. The oracle speaks in invariants; severity is a triage
#: judgement layered on top and kept in one auditable place.
SEVERITY_BY_SIGNAL: dict[str, Severity] = {
    "arithmetic": Severity.HIGH,
    "metamorphic": Severity.HIGH,
    "http": Severity.MEDIUM,
    "accessibility": Severity.MEDIUM,
    "interaction": Severity.HIGH,
}

@dataclass(frozen=True, slots=True)
class Narrative:
    """How one kind of violation is explained to a person.

    Deliberately here rather than in the oracle. The oracle decides *whether*
    an invariant broke, which is a judgement about the application; how to
    phrase that for a human, and what to do about it, is presentation. It
    changes with audience and never with evidence, and keeping it in one table
    means nobody has to read nine signal functions to find out what a report
    will say.

    ``summary`` avoids the words "invariant" and "assert". A report that only
    a test author can read is a report nobody acts on.
    """

    #: What a user would actually see, in one sentence.
    summary: str
    #: Why the application behaves this way.
    cause: str
    #: What to change to fix it.
    fix: str


#: One narrative per check, keyed by check id. A check with no entry still
#: produces a finding; it just falls back to the raw evidence.
_NARRATIVE: dict[str, Narrative] = {
    "ui_interaction": Narrative(
        summary="A control on the page does not work when it is used",
        cause=(
            "Something a visitor can click, type into or choose fails when it is "
            "actually used: the page throws a JavaScript error, a request comes "
            "back failing, the interaction hangs, or nothing at all happens. "
            "This only appears once the page is driven in a real browser, which "
            "is why no server-side check ever reports it."
        ),
        fix=(
            "Reproduce it by hand: open the page, use that control, and watch the "
            "browser console and network tab. The screenshot stored with this "
            "finding shows the page in the state the interaction left it."
        ),
    ),
    "cart_quantity_arithmetic": Narrative(
        summary="The cart charges for one item when you order several",
        cause=(
            "The line total uses the unit price on its own instead of "
            "multiplying it by the quantity ordered."
        ),
        fix=(
            "Multiply unit price by quantity for each cart line, then add "
            "those line totals up to get the subtotal."
        ),
    ),
    "rounding_precision": Narrative(
        summary="Money is cut off instead of rounded, so the shop loses a cent on some totals",
        cause=(
            "The final total drops the third decimal rather than rounding it, "
            "so any amount whose third decimal is 5 or more is charged one "
            "cent less than it should be."
        ),
        fix=(
            "Round the total to the nearest cent rather than truncating it, "
            "and round once at the end instead of on each intermediate step."
        ),
    ),
    "empty_cart_reset": Narrative(
        summary="An emptied cart still shows the total from before the items were removed",
        cause=(
            "Removing the last item clears the list of lines but never "
            "recalculates the totals, so the previous figures are left behind."
        ),
        fix=(
            "Recalculate subtotal, discount and total after every change to "
            "the cart lines, including when the last one is removed."
        ),
    ),
    "discount_idempotence": Narrative(
        summary="A discount code keeps working when applied a second time",
        cause=(
            "Each application compounds onto the existing discount instead of "
            "replacing it, so the price falls further every time someone "
            "presses Apply."
        ),
        fix=(
            "Treat the code as setting a discount rate rather than adding to "
            "one, so applying it twice leaves the price unchanged."
        ),
    ),
    "price_sort_monotonic": Narrative(
        summary='Sorting by "Price: low to high" puts items in the wrong order',
        cause=(
            "Prices are compared as text rather than as numbers, so 19.99 "
            "sorts after 149.99 because the character '1' comes before '9'."
        ),
        fix="Compare prices as numbers in the catalogue sort, not as strings.",
    ),
    "search_case_equivalence": Narrative(
        summary="Search returns different results depending on capitalisation",
        cause=(
            "The search compares the raw text, so a term only matches when the "
            "capitalisation happens to line up with the product name."
        ),
        fix="Lowercase both the search term and the product name before comparing them.",
    ),
    "pagination_disjoint": Narrative(
        summary="The same product appears on two pages of the catalogue at once",
        cause=(
            "Each page's starting position is one item too early, so "
            "consecutive pages overlap: one product is listed twice while "
            "another is never shown at all."
        ),
        fix=(
            "Start each page at (page number - 1) x items per page, with no "
            "extra adjustment."
        ),
    ),
    "negative_quantity_rejected": Narrative(
        summary="The cart accepts a negative quantity",
        cause=(
            "The add-to-cart endpoint stores whatever quantity it is sent "
            "without validating it, so a request for -1 is accepted and then "
            "pulls the total down."
        ),
        fix="Reject any quantity below 1 with a 400 response before touching the cart.",
    ),
    "referenced_routes_respond": Narrative(
        summary="Links in the page footer lead to pages that do not exist",
        cause=(
            "The footer links to routes that were never built, so following "
            "them ends in a 'page not found' error."
        ),
        fix=(
            "Either build the missing pages or remove the links from the "
            "navigation so they cannot be followed."
        ),
    ),
    "app_map_controls": Narrative(
        summary="The checkout email field has no label, so screen readers cannot say what it is for",
        cause=(
            "The field relies on placeholder text. A placeholder is a hint, "
            "not a label: it is not announced as the field's name and it "
            "disappears as soon as typing starts."
        ),
        fix=(
            "Give the input a real <label> tied to it by id, or an aria-label, "
            "that stays present and is announced."
        ),
    ),
}


def cluster(outcomes: list[SignalOutcome]) -> list[FindingDraft]:
    """Group violated signals into one draft per suspected defect."""
    by_key: dict[str, list[SignalOutcome]] = {}
    for outcome in outcomes:
        if outcome.violated is not True:
            continue
        key = outcome.suspected_bug_id or f"signal:{outcome.signal}"
        by_key.setdefault(key, []).append(outcome)

    drafts: list[FindingDraft] = []
    for key, group in sorted(by_key.items()):
        primary = max(group, key=lambda outcome: outcome.confidence)
        severity = SEVERITY_BY_SIGNAL.get(primary.signal, Severity.MEDIUM)
        narrative = _NARRATIVE.get(primary.check_id)

        # The narrative describes the *kind* of defect; the primary signal's
        # detail carries the numbers this run actually observed, so a reader
        # gets both the explanation and the evidence for it.
        title = (
            narrative.summary
            if narrative is not None
            else f"Unexpected behaviour in {primary.check_id}: {primary.detail}"
        )
        root_cause = narrative.cause if narrative is not None else primary.detail

        drafts.append(
            FindingDraft(
                key=key,
                title=title,
                severity=severity,
                root_cause=root_cause,
                suggested_fix=narrative.fix if narrative is not None else None,
                evidence={
                    "signals": [outcome.as_dict() for outcome in group],
                    "primary_signal": primary.signal,
                    "observed": primary.detail,
                },
                suspected_bug_id=None if key.startswith("signal:") else key,
            )
        )
    return drafts


async def write_findings(
    run_id: str,
    outcomes: list[SignalOutcome],
    bus: EventBus,
) -> list[Finding]:
    """Persist drafts for one run, emitting an event per finding."""
    drafts = cluster(outcomes)
    findings: list[Finding] = []
    for draft in drafts:
        finding = Finding(
            run_id=run_id,
            title=draft.title,
            severity=draft.severity,
            status=FindingStatus.OPEN,
            root_cause=draft.root_cause,
            suggested_fix=draft.suggested_fix,
            repro_steps=[draft.evidence.get("observed", draft.root_cause)],
            evidence=draft.evidence,
            matched_bug_id=draft.suspected_bug_id,
        )
        findings.append(finding)
        await bus.emit(
            EventType.FINDING_RECORDED,
            title=draft.title,
            severity=draft.severity.value,
            suspected_bug_id=draft.suspected_bug_id,
        )
    return findings


def summarize_decision(decision: VerdictDecision, outcomes: list[SignalOutcome]) -> str:
    """One-line human summary, used by the CLI run report."""
    violated = [outcome for outcome in outcomes if outcome.violated]
    if decision is VerdictDecision.BUG:
        return f"{len(violated)} invariant violation(s) detected"
    if decision is VerdictDecision.NOT_A_BUG:
        return "all exercised invariants held"
    return "evidence incomplete; no decision"
