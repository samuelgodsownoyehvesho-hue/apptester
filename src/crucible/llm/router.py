"""Provider-agnostic model routing with mandatory sensitivity enforcement.

Two invariants hold for every call made through :class:`ModelRouter`:

1. **Clearance is checked before a client exists.** A provider that is not
   cleared for the payload's data class is never selected, so a violation
   would have to be introduced here rather than at any call site.
2. **Escalation is upward only.** When the preferred provider for a tier is
   not cleared, routing moves to a *more* constrained provider, never a less
   constrained one. There is no code path that downgrades clearance to keep a
   call cheap.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ValidationError

from crucible.core.budget import RunBudget
from crucible.core.config import Settings
from crucible.core.logging import get_logger
from crucible.llm.ledger import CostLedger, LLMSpan, estimate_cost
from crucible.llm.providers import CompletionResult, ModelClient, RateLimiter
from crucible.llm.sensitivity import (
    DataClass,
    ProviderName,
    SensitivityViolation,
    is_cleared,
)

logger = get_logger(__name__)

#: Rough characters-per-token ratio for pre-flight budget estimation. English
#: prose sits near 4; code and JSON are closer to 3, so this stays pessimistic.
CHARS_PER_TOKEN = 3.5


class Tier(StrEnum):
    """Which quality/cost band a call belongs to."""

    #: High-volume, low-stakes work: crawling, summarization, test generation.
    CHEAP = "cheap"
    #: Low-volume, high-stakes work: verdicts, adjudication, root cause.
    FRONTIER = "frontier"
    #: Local inference. Plumbing only; never authorises a verdict.
    LOCAL = "local"


#: Providers to try, in order, for each tier.
#:
#: Ordering is a quality and cost decision. Clearance is applied afterwards as
#: a filter, which is what makes escalation upward-only by construction: an
#: uncleared provider is skipped rather than substituted.
TIER_PREFERENCE: dict[Tier, tuple[ProviderName, ...]] = {
    Tier.CHEAP: (ProviderName.GEMINI, ProviderName.NVIDIA, ProviderName.OLLAMA),
    # Gemini is deliberately absent. Silently judging with a different model
    # would change what a verdict means, so an unavailable judge is an error.
    Tier.FRONTIER: (ProviderName.NVIDIA, ProviderName.OLLAMA),
    Tier.LOCAL: (ProviderName.OLLAMA,),
}

#: Tiers permitted to authorise a bug verdict. Local inference is excluded:
#: a small model's disagreement is not evidence about the software under test.
VERDICT_AUTHORISED_TIERS = frozenset({Tier.FRONTIER})


class NoProviderAvailable(RuntimeError):
    """Raised when no configured provider is cleared for a request."""

    def __init__(self, tier: Tier, data_class: DataClass, available: tuple[ProviderName, ...]) -> None:
        candidates = ", ".join(p.value for p in TIER_PREFERENCE[tier]) or "none"
        configured = ", ".join(p.value for p in available) or "none"
        super().__init__(
            f"No provider available for tier={tier.value!r} with {data_class.value!r} "
            f"clearance. Candidates for this tier: {candidates}. Configured providers: "
            f"{configured}. Check API keys and ENABLE_* flags in .env."
        )


@dataclass(frozen=True, slots=True)
class Selection:
    """The outcome of routing a request to a provider."""

    provider: ProviderName
    tier: Tier
    #: True when the tier's first choice was skipped for clearance reasons.
    escalated: bool

    def __str__(self) -> str:
        suffix = " (escalated)" if self.escalated else ""
        return f"{self.provider.value}/{self.tier.value}{suffix}"


class ModelRouter:
    """Routes completions to providers, enforcing clearance and budgets."""

    def __init__(
        self,
        settings: Settings,
        ledger: CostLedger,
        *,
        budget: RunBudget | None = None,
        run_id: str | None = None,
    ) -> None:
        self._settings = settings
        self._ledger = ledger
        self._budget = budget
        self._run_id = run_id
        self._clients: dict[ProviderName, ModelClient] = {}

    # -- routing ---------------------------------------------------------

    def select(self, tier: Tier, data_class: DataClass) -> Selection:
        """Choose a provider for ``tier`` that is cleared for ``data_class``.

        Raises :class:`NoProviderAvailable` if nothing qualifies, and
        :class:`SensitivityViolation` for a ``SECRET`` payload.
        """
        if data_class is DataClass.SECRET:
            raise SensitivityViolation(
                ProviderName.GEMINI, data_class, "No provider may receive credentials."
            )

        # Escalation means a provider we *could* have used was passed over for
        # clearance reasons. A provider that is simply unconfigured was never a
        # candidate, so counting it as escalation would inflate the metric and
        # mask the real signal.
        skipped_for_clearance = False
        for provider in TIER_PREFERENCE[tier]:
            if self._settings.provider_config(provider) is None:
                continue
            if not is_cleared(provider, data_class):
                logger.debug(
                    "clearance_skip provider=%s data_class=%s tier=%s",
                    provider.value, data_class.value, tier.value,
                )
                skipped_for_clearance = True
                continue
            return Selection(
                provider=provider, tier=tier, escalated=skipped_for_clearance
            )

        raise NoProviderAvailable(tier, data_class, self._settings.available_providers())

    def _client_for(self, provider: ProviderName) -> ModelClient:
        """Return a cached client for ``provider``."""
        if provider not in self._clients:
            config = self._settings.provider_config(provider)
            if config is None:
                raise NoProviderAvailable(Tier.CHEAP, DataClass.PUBLIC, ())
            self._clients[provider] = ModelClient(
                config,
                rate_limiter=RateLimiter(self._settings.max_requests_per_minute),
            )
        return self._clients[provider]

    # -- completions -----------------------------------------------------

    async def complete(
        self,
        *,
        tier: Tier,
        data_class: DataClass,
        agent: str,
        messages: list[dict[str, str]],
        json_mode: bool = False,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        case_id: str | None = None,
    ) -> CompletionResult:
        """Run a completion through the full guard chain.

        The guard chain runs in a fixed order: select a cleared provider,
        pre-flight the budget, issue the call, then record the actual spend.
        """
        selection = self.select(tier, data_class)
        client = self._client_for(selection.provider)

        # Pre-flight so an over-budget call is never dispatched.
        estimated_prompt = int(sum(len(m.get("content", "")) for m in messages) / CHARS_PER_TOKEN)
        estimated_total = estimated_prompt + (max_tokens or 1024)
        self._preflight(cost=estimate_cost(client.config.model, estimated_prompt, max_tokens or 1024),
                        tokens=estimated_total)

        try:
            result = await client.complete(
                messages,
                json_mode=json_mode,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            self._record(
                selection, client.config.model, data_class, agent,
                prompt_tokens=0, completion_tokens=0, latency_ms=0.0,
                ok=False, error=f"{type(exc).__name__}: {exc}", case_id=case_id,
            )
            raise

        cost = estimate_cost(result.model, result.prompt_tokens, result.completion_tokens)
        self._record(
            selection, result.model, data_class, agent,
            prompt_tokens=result.prompt_tokens, completion_tokens=result.completion_tokens,
            latency_ms=result.latency_ms, ok=True, attempts=result.attempts, case_id=case_id,
        )
        if self._budget is not None:
            self._budget.add(cost_usd=cost, tokens=result.total_tokens)

        return result

    async def complete_structured[ModelT: BaseModel](
        self,
        *,
        tier: Tier,
        data_class: DataClass,
        agent: str,
        messages: list[dict[str, str]],
        response_model: type[ModelT],
        max_repairs: int = 2,
        max_tokens: int | None = None,
        case_id: str | None = None,
    ) -> ModelT:
        """Run a completion and coerce it into ``response_model``.

        JSON mode plus a schema description is used rather than provider-native
        structured output, because support for the latter varies across free
        tiers. Validation failures are fed back to the model as a repair
        request, which recovers most malformed responses in one extra call.
        """
        schema = json.dumps(response_model.model_json_schema(), indent=None)
        instruction = (
            "Respond with a single JSON object and nothing else. It must validate "
            f"against this JSON Schema:\n{schema}"
        )
        working = [*messages]
        if working and working[0].get("role") == "system":
            working[0] = {
                "role": "system",
                "content": f"{working[0]['content']}\n\n{instruction}",
            }
        else:
            working.insert(0, {"role": "system", "content": instruction})

        last_error: ValidationError | None = None
        for attempt in range(max_repairs + 1):
            result = await self.complete(
                tier=tier,
                data_class=data_class,
                agent=agent,
                messages=working,
                json_mode=True,
                max_tokens=max_tokens,
                case_id=case_id,
            )
            try:
                return response_model.model_validate_json(result.text)
            except ValidationError as exc:
                last_error = exc
                logger.warning(
                    "structured_repair agent=%s attempt=%d errors=%d",
                    agent, attempt + 1, exc.error_count(),
                )
                working = [
                    *working,
                    {"role": "assistant", "content": result.text},
                    {
                        "role": "user",
                        "content": (
                            "That response failed schema validation with these errors:\n"
                            f"{exc}\n\nReturn a corrected JSON object only."
                        ),
                    },
                ]

        assert last_error is not None
        raise last_error

    # -- internals -------------------------------------------------------

    def _preflight(self, *, cost: float, tokens: int) -> None:
        if self._budget is not None:
            self._budget.check(cost_usd=cost, tokens=tokens)

    def _record(
        self,
        selection: Selection,
        model: str,
        data_class: DataClass,
        agent: str,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        ok: bool,
        attempts: int = 1,
        error: str | None = None,
        case_id: str | None = None,
    ) -> None:
        span = LLMSpan(
            provider=selection.provider,
            model=model,
            tier=selection.tier.value,
            data_class=data_class,
            agent=agent,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=estimate_cost(model, prompt_tokens, completion_tokens),
            latency_ms=latency_ms,
            ok=ok,
            run_id=self._run_id,
            case_id=case_id,
            escalated=selection.escalated,
            attempts=attempts,
            error=error,
        )
        self._ledger.record(span)
        logger.debug("llm_span %s", json.dumps(span.as_dict(), default=str))

    async def aclose(self) -> None:
        """Close every pooled client."""
        for client in self._clients.values():
            await client.aclose()
        self._clients.clear()

    def diagnostics(self) -> dict[str, Any]:
        """Report which providers are usable and what each tier resolves to."""
        available = self._settings.available_providers()
        tiers: dict[str, Any] = {}
        for tier in Tier:
            tiers[tier.value] = {}
            for data_class in (DataClass.PUBLIC, DataClass.PROPRIETARY):
                try:
                    tiers[tier.value][data_class.value] = str(self.select(tier, data_class))
                except NoProviderAvailable:
                    tiers[tier.value][data_class.value] = "unavailable"
        return {
            "configured_providers": [p.value for p in available],
            "tiers": tiers,
            "budget": self._budget.snapshot() if self._budget else None,
        }
