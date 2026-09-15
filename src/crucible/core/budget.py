"""Run budgets: hard stops on spend and token consumption.

Free tiers make this module load-bearing rather than nice-to-have. A single
provider cap of 200k tokens/day is roughly two or three unguarded agent runs,
so a run that cannot stop itself cannot be iterated on at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class BudgetExceeded(RuntimeError):
    """Raised when a run would exceed its cost or token ceiling.

    Aborting is the intended behaviour: a run that overshoots silently produces
    a bill, and on a free tier it produces a cascade of 429s that corrupts
    results instead of failing cleanly.
    """

    def __init__(self, resource: str, limit: float, used: float, requested: float = 0.0) -> None:
        self.resource = resource
        self.limit = limit
        self.used = used
        self.requested = requested
        super().__init__(
            f"Run budget exceeded for {resource}: used {used:,.2f} of {limit:,.2f}, "
            f"next call needs {requested:,.2f}."
        )


@dataclass
class RunBudget:
    """Tracks consumption for a single run and refuses to overshoot."""

    max_cost_usd: float
    max_tokens: int
    spent_usd: float = 0.0
    tokens_used: int = 0
    calls: int = 0
    #: Set once a call has pushed usage past a ceiling, so subsequent checks
    #: refuse even if the overshoot was only just noticed.
    _exhausted: bool = field(default=False, repr=False)

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.max_cost_usd - self.spent_usd)

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.max_tokens - self.tokens_used)

    @property
    def exhausted(self) -> bool:
        return self._exhausted

    def check(self, *, cost_usd: float = 0.0, tokens: int = 0) -> None:
        """Raise :class:`BudgetExceeded` if a call of this size cannot be afforded.

        Call this *before* dispatching a request, using a conservative
        estimate, so the overshoot is never actually incurred.
        """
        if self._exhausted:
            raise BudgetExceeded("run", 1.0, 1.0)

        if self.spent_usd + cost_usd > self.max_cost_usd:
            raise BudgetExceeded("cost_usd", self.max_cost_usd, self.spent_usd, cost_usd)

        if self.tokens_used + tokens > self.max_tokens:
            raise BudgetExceeded("tokens", float(self.max_tokens), float(self.tokens_used), float(tokens))

    def add(self, *, cost_usd: float, tokens: int) -> None:
        """Record actual consumption after a call returns."""
        self.spent_usd += cost_usd
        self.tokens_used += tokens
        self.calls += 1

        if self.spent_usd > self.max_cost_usd or self.tokens_used > self.max_tokens:
            # The estimate was wrong. Latch so the next call stops instead.
            self._exhausted = True

    def snapshot(self) -> dict[str, float | int]:
        """Return a summary suitable for logging or persisting on the run row."""
        return {
            "calls": self.calls,
            "spent_usd": round(self.spent_usd, 6),
            "max_cost_usd": self.max_cost_usd,
            "tokens_used": self.tokens_used,
            "max_tokens": self.max_tokens,
            "remaining_usd": round(self.remaining_usd, 6),
            "remaining_tokens": self.remaining_tokens,
        }
