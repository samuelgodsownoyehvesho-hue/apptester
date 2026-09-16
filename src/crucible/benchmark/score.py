"""Benchmark: score a run against the target's ground truth.

The scoring contract is deliberately strict, because a generous scorer would
flatter the agent and every downstream number would be fiction:

* A declared defect **counts as found** only when a finding carries its id
  *and* that finding's evidence contains a violated signal.
* A finding whose suspected id is not actually active in the target scores as
  a false positive — the label is a claim, verified against the manifest.
* A declared defect with no matching finding is a miss. There is no partial
  credit and no "close enough".

Ground truth is read from the target itself (``/api/ground-truth/bugs``), not
hardcoded here, so the scorer works against any mutant selection without
edits. That endpoint is deliberately unlinked from the target's UI; the
harness must fetch it explicitly, exactly because an exploring agent should
never find it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from crucible.execute.client import AppClient
from crucible.oracle.signals import SignalOutcome
from crucible.store.models import Finding
from crucible.triage.cluster import cluster

BENCHMARK_MANIFEST_PATH = "/api/ground-truth/bugs"


@dataclass(frozen=True, slots=True)
class Manifest:
    """The active ground truth reported by the target itself."""

    all_ids: frozenset[str]
    active_ids: frozenset[str]
    selection: str

    @classmethod
    def from_json(cls, payload: Any) -> Manifest:
        if not isinstance(payload, dict):
            raise ValueError("benchmark manifest was not a JSON object")
        all_ids = payload.get("allIds")
        active_ids = payload.get("activeIds")
        if not isinstance(all_ids, list) or not isinstance(active_ids, list):
            raise ValueError("benchmark manifest is missing id lists")
        return cls(
            all_ids=frozenset(str(item) for item in all_ids),
            active_ids=frozenset(str(item) for item in active_ids),
            selection=str(payload.get("selection", "")),
        )


@dataclass(slots=True)
class ScoreReport:
    """Recall, precision, and the pairings that produced them."""

    declared_active: int
    found: int = 0
    missed: list[str] = field(default_factory=list)
    true_positives: list[str] = field(default_factory=list)
    false_positives: list[str] = field(default_factory=list)

    @property
    def recall(self) -> float:
        denominator = self.declared_active
        return self.found / denominator if denominator else 1.0

    @property
    def precision(self) -> float:
        reported = len(self.true_positives) + len(self.false_positives)
        return len(self.true_positives) / reported if reported else 1.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "declared_active": self.declared_active,
            "found": self.found,
            "recall": round(self.recall, 4),
            "precision": round(self.precision, 4),
            "missed": self.missed,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
        }


async def fetch_manifest(client: AppClient) -> Manifest:
    """Read the ground-truth manifest from the target."""
    response = await client.get(BENCHMARK_MANIFEST_PATH)
    if response.error or not response.is_json:
        raise ValueError(
            f"benchmark manifest unavailable at {BENCHMARK_MANIFEST_PATH}: "
            f"{response.error or response.status}"
        )
    return Manifest.from_json(response.json_body)


def score(
    manifest: Manifest,
    findings: list[Finding],
    outcomes: list[SignalOutcome],
) -> ScoreReport:
    """Pair findings against the active defect set.

    ``outcomes`` re-enters the cluster step so evidence can be attributed per
    finding without trusting persisted JSON to have the right shape.
    """
    drafts = {draft.key: draft for draft in cluster(outcomes)}

    violated_by_bug: dict[str, bool] = {}
    for draft in drafts.values():
        if draft.suspected_bug_id is None:
            continue
        signals = draft.evidence.get("signals", [])
        violated_by_bug[draft.suspected_bug_id] = any(
            isinstance(signal, dict) and signal.get("violated") is True
            for signal in signals
        )

    reported_ids = {
        finding.matched_bug_id
        for finding in findings
        if finding.matched_bug_id is not None
    }

    report = ScoreReport(declared_active=len(manifest.active_ids))
    for bug_id in sorted(manifest.active_ids):
        if bug_id in reported_ids and violated_by_bug.get(bug_id, False):
            report.true_positives.append(bug_id)
        else:
            report.missed.append(bug_id)

    report.false_positives = sorted(
        bug_id for bug_id in reported_ids if bug_id not in manifest.active_ids
    )
    # Found means: real, verified hits. A match without a violated signal is a
    # miss, not a half-hit.
    report.found = len(report.true_positives)
    report.missed.sort()
    return report
