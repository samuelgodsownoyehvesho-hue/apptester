"""Central configuration, loaded from the environment and ``.env``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

from crucible.llm.sensitivity import ProviderName


class ProviderConfig:
    """Resolved connection details for a single provider."""

    __slots__ = ("api_key", "base_url", "model", "name")

    def __init__(self, name: ProviderName, base_url: str, model: str, api_key: str) -> None:
        self.name = name
        self.base_url = base_url
        self.model = model
        self.api_key = api_key

    def __repr__(self) -> str:  # never leak the key into logs or tracebacks
        return (
            f"ProviderConfig(name={self.name.value!r}, base_url={self.base_url!r}, "
            f"model={self.model!r}, api_key='***')"
        )


class Settings(BaseSettings):
    """Process-wide settings.

    Every provider is described by three fields plus an enable flag, so that
    swapping a provider is configuration rather than code. Free tiers churn
    monthly; the router must survive that without a refactor.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Tier 1: Google Gemini (PUBLIC data only) ---
    gemini_api_key: SecretStr = SecretStr("")
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    gemini_model: str = "gemini-flash-latest"
    enable_gemini: bool = True

    # --- Tier 2: NVIDIA NIM (PUBLIC + PROPRIETARY) ---
    nvidia_api_key: SecretStr = SecretStr("")
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    # Model ids on this catalogue are retired on a schedule: the previous default
    # (meta/llama-3.3-70b-instruct) was withdrawn and now answers every request
    # with 410 Gone. Verify against {nvidia_base_url}/models rather than trusting
    # this name, and treat a 404 "not found for account" as "not entitled".
    nvidia_model: str = "meta/llama-3.2-11b-vision-instruct"
    enable_nvidia: bool = True

    # --- Tier 3: Ollama (local, optional) ---
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "qwen2.5:7b"
    enable_ollama: bool = False

    # --- Interaction lane: press everything reconnaissance found ---
    interact_enabled: bool = True
    #: 0 means no ceiling. Coverage is the point of the lane, so the default
    #: exercises every element found; set a number only to bound a run on
    #: purpose, because a silently truncated pass reads as coverage it is not.
    interact_max_elements: int = 0
    #: A control that hangs must fail its own check, never stall the others.
    interact_timeout_ms: int = Field(default=15_000, gt=0)
    interact_settle_ms: int = Field(default=500, ge=0)
    #: Spacing between interactions, so a thorough pass does not trip the rate
    #: limiter of a real deployed site and lose the rest of the run to it.
    interact_delay_ms: int = Field(default=150, ge=0)

    # --- Budgets: hard stops, not warnings ---
    max_run_cost_usd: float = Field(default=2.00, gt=0)
    max_run_tokens: int = Field(default=2_000_000, gt=0)
    max_requests_per_minute: int = Field(default=30, gt=0)

    # --- Storage ---
    crucible_db_url: str = "sqlite:///./crucible.db"
    crucible_artifacts_dir: Path = Path("./artifacts")

    # --- Safety ---
    allow_production_targets: bool = False
    egress_allowlist: str = ""

    # --- Logging ---
    log_level: str = "INFO"
    log_json: bool = False

    @computed_field  # type: ignore[prop-decorator]
    @property
    def egress_hosts(self) -> tuple[str, ...]:
        """Parse the comma-separated egress allowlist into unique hostnames."""
        hosts = (h.strip().lower() for h in self.egress_allowlist.split(","))
        return tuple(sorted({h for h in hosts if h}))

    def provider_config(self, name: ProviderName) -> ProviderConfig | None:
        """Return connection details for ``name``, or ``None`` if unusable.

        Returns ``None`` when the provider is disabled or has no key, which
        lets callers degrade gracefully instead of crashing at import time.
        """
        match name:
            case ProviderName.GEMINI:
                if not self.enable_gemini:
                    return None
                key = self.gemini_api_key.get_secret_value()
                if not key:
                    return None
                return ProviderConfig(name, self.gemini_base_url, self.gemini_model, key)

            case ProviderName.NVIDIA:
                if not self.enable_nvidia:
                    return None
                key = self.nvidia_api_key.get_secret_value()
                if not key:
                    return None
                return ProviderConfig(name, self.nvidia_base_url, self.nvidia_model, key)

            case ProviderName.OLLAMA:
                if not self.enable_ollama:
                    return None
                # A local runtime needs no credential; a placeholder keeps the
                # OpenAI-compatible client happy without implying a secret.
                return ProviderConfig(
                    name, self.ollama_base_url, self.ollama_model, api_key="ollama"
                )

        return None

    def available_providers(self) -> tuple[ProviderName, ...]:
        """Every provider that is currently usable."""
        return tuple(
            name for name in ProviderName if self.provider_config(name) is not None
        )

    def missing_provider_hint(self) -> str:
        """Human-readable guidance when no provider is configured."""
        return (
            "No model provider is configured. Set GEMINI_API_KEY (tier 1) and/or "
            "NVIDIA_API_KEY (tier 2) in .env. See .env.example."
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached process-wide settings."""
    return Settings()
