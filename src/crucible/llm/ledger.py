"""Cost and token accounting for every model call.

Every completion produces exactly one :class:`LLMSpan`. Attribution matters
more than the total: knowing a run cost $0.40 is useless, knowing which test
case spent it is what lets you prune the plan.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from crucible.llm.sensitivity import DataClass, ProviderName

#: USD per million tokens, as ``(input, output)``.
#:
#: Free tiers cost nothing, so these values are *notional*: they record what
#: the same call would cost on a paid plan. Two reasons to keep them. Budgets
#: stay meaningful after a migration to a paid provider, and comparing the
#: notional cost of two strategies is how we decide whether an agent is worth
#: its tokens. Values are approximate and should be revisited whenever a
#: provider changes its pricing.
PRICING: dict[str, tuple[float, float]] = {
    # Tier 1 (free tier in use; priced as the paid equivalent)
    "gemini-flash": (0.10, 0.40),
    "gemini-pro": (1.25, 5.00),
    # Tier 2 (prototyping access)
    "meta/llama": (0.20, 0.20),
    "nvidia/": (0.20, 0.20),
    "deepseek": (0.15, 0.60),
    "qwen": (0.10, 0.10),
    # Tier 3 (local — electricity, not dollars)
    "ollama": (0.0, 0.0),
}

#: Fallback rate when a model is not recognised. Deliberately pessimistic:
#: an unknown model should look expensive so it gets noticed.
DEFAULT_PRICING: tuple[float, float] = (1.00, 3.00)


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Return the notional USD cost of a call."""
    rate_in, rate_out = DEFAULT_PRICING
    lowered = model.lower()
    for prefix, rates in PRICING.items():
        if prefix in lowered:
            rate_in, rate_out = rates
            break
    return (prompt_tokens / 1_000_000) * rate_in + (completion_tokens / 1_000_000) * rate_out


@dataclass(frozen=True, slots=True)
class LLMSpan:
    """One completed (or failed) model call."""

    provider: ProviderName
    model: str
    tier: str
    data_class: DataClass
    agent: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: float
    ok: bool
    run_id: str | None = None
    case_id: str | None = None
    escalated: bool = False
    attempts: int = 1
    error: str | None = None
    ts: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_dict(self) -> dict[str, Any]:
        return {
            "ts": self.ts.isoformat(),
            "run_id": self.run_id,
            "case_id": self.case_id,
            "provider": self.provider.value,
            "model": self.model,
            "tier": self.tier,
            "data_class": self.data_class.value,
            "agent": self.agent,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "latency_ms": round(self.latency_ms, 1),
            "ok": self.ok,
            "escalated": self.escalated,
            "attempts": self.attempts,
            "error": self.error,
        }


class CostLedger:
    """Collects spans and reports aggregates.

    The ``on_record`` hook is how persistence attaches later without this
    module importing the storage layer.
    """

    def __init__(self, on_record: Callable[[LLMSpan], None] | None = None) -> None:
        self._spans: list[LLMSpan] = []
        self._on_record = on_record

    def record(self, span: LLMSpan) -> None:
        self._spans.append(span)
        if self._on_record is not None:
            self._on_record(span)

    @property
    def spans(self) -> tuple[LLMSpan, ...]:
        return tuple(self._spans)

    @property
    def total_cost_usd(self) -> float:
        return sum(span.cost_usd for span in self._spans)

    @property
    def total_tokens(self) -> int:
        return sum(span.total_tokens for span in self._spans)

    @property
    def escalations(self) -> int:
        """Calls that were pushed to a higher tier for clearance reasons.

        Worth watching: sustained escalation means a tier-1 provider is
        unusable for the data being processed, and the high-volume tier-2
        budget will not survive it.
        """
        return sum(1 for span in self._spans if span.escalated)

    def by_agent(self) -> dict[str, dict[str, float | int]]:
        """Spend and tokens grouped by the agent that incurred them."""
        buckets: dict[str, dict[str, float | int]] = defaultdict(
            lambda: {"calls": 0, "cost_usd": 0.0, "tokens": 0}
        )
        for span in self._spans:
            bucket = buckets[span.agent]
            bucket["calls"] += 1
            bucket["cost_usd"] = float(bucket["cost_usd"]) + span.cost_usd
            bucket["tokens"] = int(bucket["tokens"]) + span.total_tokens
        return {agent: dict(values) for agent, values in buckets.items()}

    def summary(self) -> dict[str, Any]:
        """Return an aggregate suitable for the run record or CLI output."""
        by_provider: dict[str, int] = defaultdict(int)
        for span in self._spans:
            by_provider[span.provider.value] += 1

        return {
            "calls": len(self._spans),
            "failures": sum(1 for span in self._spans if not span.ok),
            "escalations": self.escalations,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_tokens": self.total_tokens,
            "calls_by_provider": dict(by_provider),
            "by_agent": self.by_agent(),
        }
