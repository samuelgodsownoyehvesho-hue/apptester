"""Budget hard-stop behaviour.

A budget that only warns is not a budget. These tests assert that an
overspending run stops rather than continuing to emit 429s.
"""

from __future__ import annotations

import pytest

from crucible.core.budget import BudgetExceeded, RunBudget


class TestPreflight:
    def test_allows_call_within_limits(self) -> None:
        budget = RunBudget(max_cost_usd=1.0, max_tokens=1000)
        budget.check(cost_usd=0.10, tokens=100)

    def test_rejects_call_exceeding_cost(self) -> None:
        budget = RunBudget(max_cost_usd=1.0, max_tokens=1000)
        with pytest.raises(BudgetExceeded) as exc_info:
            budget.check(cost_usd=1.50, tokens=10)
        assert exc_info.value.resource == "cost_usd"

    def test_rejects_call_exceeding_tokens(self) -> None:
        budget = RunBudget(max_cost_usd=10.0, max_tokens=1000)
        with pytest.raises(BudgetExceeded) as exc_info:
            budget.check(cost_usd=0.01, tokens=1001)
        assert exc_info.value.resource == "tokens"

    def test_accumulated_spend_counts_toward_limit(self) -> None:
        budget = RunBudget(max_cost_usd=1.0, max_tokens=10_000)
        budget.add(cost_usd=0.60, tokens=500)
        budget.add(cost_usd=0.30, tokens=500)
        with pytest.raises(BudgetExceeded):
            budget.check(cost_usd=0.20, tokens=100)


class TestLatching:
    def test_overshoot_latches_exhausted(self) -> None:
        budget = RunBudget(max_cost_usd=1.0, max_tokens=1000)
        # Simulate an estimate that was too low; actual spend overshoots.
        budget.add(cost_usd=1.40, tokens=100)
        assert budget.exhausted

    def test_latched_budget_refuses_even_free_calls(self) -> None:
        budget = RunBudget(max_cost_usd=1.0, max_tokens=1000)
        budget.add(cost_usd=1.40, tokens=100)
        with pytest.raises(BudgetExceeded):
            budget.check(cost_usd=0.0, tokens=0)

    def test_token_overshoot_also_latches(self) -> None:
        budget = RunBudget(max_cost_usd=100.0, max_tokens=1000)
        budget.add(cost_usd=0.01, tokens=5000)
        assert budget.exhausted


class TestReporting:
    def test_remaining_never_goes_negative(self) -> None:
        budget = RunBudget(max_cost_usd=1.0, max_tokens=100)
        budget.add(cost_usd=5.0, tokens=500)
        assert budget.remaining_usd == 0.0
        assert budget.remaining_tokens == 0

    def test_add_tracks_calls(self) -> None:
        budget = RunBudget(max_cost_usd=1.0, max_tokens=100)
        budget.add(cost_usd=0.01, tokens=10)
        budget.add(cost_usd=0.01, tokens=10)
        assert budget.calls == 2
        assert budget.spent_usd == pytest.approx(0.02)
        assert budget.tokens_used == 20

    def test_snapshot_reports_headroom(self) -> None:
        budget = RunBudget(max_cost_usd=2.0, max_tokens=1000)
        budget.add(cost_usd=0.50, tokens=250)
        snapshot = budget.snapshot()
        assert snapshot["spent_usd"] == pytest.approx(0.5)
        assert snapshot["tokens_used"] == 250
        assert snapshot["remaining_usd"] == pytest.approx(1.5)
        assert snapshot["remaining_tokens"] == 750


class TestConstruction:
    @pytest.mark.parametrize(("cost", "tokens"), [(0.0, 100), (-1.0, 100), (1.0, 0)])
    def test_rejects_nonsensical_limits(self, cost: float, tokens: int) -> None:
        from pydantic import ValidationError

        from crucible.core.config import Settings

        with pytest.raises(ValidationError):
            Settings(
                _env_file=None,
                max_run_cost_usd=cost,
                max_run_tokens=tokens,
                gemini_api_key="",
                nvidia_api_key="",
            )
