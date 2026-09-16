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


#: Signal -> severity. The oracle speaks in invariants; severity is a triage
#: judgement layered on top and kept in one auditable place.
SEVERITY_BY_SIGNAL: dict[str, Severity] = {
    "arithmetic": Severity.HIGH,
    "metamorphic": Severity.HIGH,
    "http": Severity.MEDIUM,
    "accessibility": Severity.MEDIUM,
}

#: A per-check fallback title when a signal contributes to a shared finding.
_CHECK_LABEL: dict[str, str] = {
    "cart_quantity_arithmetic": "cart quantity arithmetic",
    "empty_cart_reset": "empty-cart total reset",
    "discount_idempotence": "discount code idempotence",
    "price_sort_monotonic": "price sort ordering",
    "search_case_equivalence": "search case equivalence",
    "pagination_disjoint": "pagination disjointness",
    "negative_quantity_rejected": "negative quantity rejection",
    "referenced_routes_respond": "linked route availability",
    "app_map_controls": "accessible names on form controls",
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
        label = _CHECK_LABEL.get(primary.check_id, primary.check_id)

        if key.startswith("signal:"):
            title = f"Invariant violated: {label}"
        else:
            title = f"{key}: {label} contradicts the declared invariant"

        drafts.append(
            FindingDraft(
                key=key,
                title=title,
                severity=severity,
                root_cause=primary.detail,
                evidence={
                    "signals": [outcome.as_dict() for outcome in group],
                    "primary_signal": primary.signal,
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
            repro_steps=[draft.root_cause],
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
