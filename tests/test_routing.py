"""Routing invariants.

The load-bearing claims here are that routing never downgrades clearance and
never silently substitutes a different model for the judge. Both would be easy
to break in a refactor and both would be invisible in production.
"""

from __future__ import annotations

import pytest

from crucible.core.config import Settings
from crucible.llm.ledger import CostLedger
from crucible.llm.router import (
    TIER_PREFERENCE,
    ModelRouter,
    NoProviderAvailable,
    Tier,
)
from crucible.llm.sensitivity import DataClass, ProviderName, SensitivityViolation


@pytest.fixture
def router_factory():
    def _make(settings: Settings) -> ModelRouter:
        return ModelRouter(settings, CostLedger(), run_id="test-run")

    return _make


class TestPreferenceOrder:
    def test_frontier_tier_never_lists_gemini(self) -> None:
        # Judging with a different model changes what a verdict means, so the
        # judge tier must not fall back to tier 1.
        assert ProviderName.GEMINI not in TIER_PREFERENCE[Tier.FRONTIER]

    def test_cheap_tier_prefers_gemini(self) -> None:
        assert TIER_PREFERENCE[Tier.CHEAP][0] is ProviderName.GEMINI


class TestSelect:
    def test_public_work_uses_tier_one(
        self, router_factory, gemini_and_nvidia: Settings
    ) -> None:
        selection = router_factory(gemini_and_nvidia).select(Tier.CHEAP, DataClass.PUBLIC)
        assert selection.provider is ProviderName.GEMINI
        assert not selection.escalated

    def test_proprietary_work_escalates_off_tier_one(
        self, router_factory, gemini_and_nvidia: Settings
    ) -> None:
        selection = router_factory(gemini_and_nvidia).select(
            Tier.CHEAP, DataClass.PROPRIETARY
        )
        assert selection.provider is ProviderName.NVIDIA
        assert selection.escalated

    def test_frontier_work_uses_tier_two(
        self, router_factory, gemini_and_nvidia: Settings
    ) -> None:
        selection = router_factory(gemini_and_nvidia).select(
            Tier.FRONTIER, DataClass.PUBLIC
        )
        assert selection.provider is ProviderName.NVIDIA

    def test_local_tier_uses_ollama_when_enabled(self, router_factory) -> None:
        settings = Settings(
            _env_file=None,
            gemini_api_key="",
            nvidia_api_key="",
            enable_gemini=False,
            enable_nvidia=False,
            enable_ollama=True,
        )
        selection = router_factory(settings).select(Tier.LOCAL, DataClass.PROPRIETARY)
        assert selection.provider is ProviderName.OLLAMA


class TestFailClosed:
    def test_proprietary_work_fails_without_a_cleared_provider(
        self, router_factory, gemini_only: Settings
    ) -> None:
        # Gemini is the only provider and it is not cleared. Routing must
        # refuse rather than downgrade the data class.
        with pytest.raises(NoProviderAvailable):
            router_factory(gemini_only).select(Tier.CHEAP, DataClass.PROPRIETARY)

    def test_frontier_work_fails_rather_than_downgrading_to_tier_one(
        self, router_factory, gemini_only: Settings
    ) -> None:
        # The critical case: a configured, cleared provider exists (Gemini for
        # PUBLIC), but it is not an acceptable judge. Substituting it would
        # silently weaken every verdict while appearing to work.
        with pytest.raises(NoProviderAvailable):
            router_factory(gemini_only).select(Tier.FRONTIER, DataClass.PUBLIC)

    def test_secrets_are_refused_by_every_tier(
        self, router_factory, gemini_and_nvidia: Settings
    ) -> None:
        router = router_factory(gemini_and_nvidia)
        for tier in Tier:
            with pytest.raises(SensitivityViolation):
                router.select(tier, DataClass.SECRET)

    def test_no_providers_configured_fails(self, router_factory, settings_factory) -> None:
        with pytest.raises(NoProviderAvailable):
            router_factory(settings_factory()).select(Tier.CHEAP, DataClass.PUBLIC)

    def test_error_message_names_the_configured_providers(
        self, router_factory, gemini_only: Settings
    ) -> None:
        with pytest.raises(NoProviderAvailable, match="gemini"):
            router_factory(gemini_only).select(Tier.FRONTIER, DataClass.PUBLIC)


class TestNvidiaOnly:
    def test_proprietary_cheap_work_goes_to_nvidia_not_escalated(
        self, router_factory, nvidia_only: Settings
    ) -> None:
        # Nothing was skipped, so this is a normal route rather than an
        # escalation. Conflating the two would hide real escalation pressure.
        selection = router_factory(nvidia_only).select(Tier.CHEAP, DataClass.PROPRIETARY)
        assert selection.provider is ProviderName.NVIDIA
        assert not selection.escalated


class TestDiagnostics:
    def test_diagnostics_reports_unavailable_tiers(
        self, router_factory, gemini_only: Settings
    ) -> None:
        report = router_factory(gemini_only).diagnostics()
        assert report["configured_providers"] == ["gemini"]
        assert report["tiers"]["cheap"]["public"] == "gemini/cheap"
        assert report["tiers"]["cheap"]["proprietary"] == "unavailable"
        assert report["tiers"]["frontier"]["public"] == "unavailable"

    def test_diagnostics_marks_escalation(
        self, router_factory, gemini_and_nvidia: Settings
    ) -> None:
        report = router_factory(gemini_and_nvidia).diagnostics()
        assert report["tiers"]["cheap"]["proprietary"] == "nvidia/cheap (escalated)"

    def test_diagnostics_needs_no_network(
        self, router_factory, gemini_and_nvidia: Settings
    ) -> None:
        # Selection is pure configuration resolution; building a client or
        # issuing a request must not be required to report routing.
        report = router_factory(gemini_and_nvidia).diagnostics()
        assert report["budget"] is None
