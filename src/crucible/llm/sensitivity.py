"""Data classification and provider clearance.

This module is the *only* place that decides whether a payload is allowed to
reach a given model provider. Every LLM call in the codebase routes through
``require_clearance`` before a client is constructed.

The rule that matters: free-tier providers may be used for non-sensitive
development data, but a payload derived from a real repository, a real
application, or real customer records must never be sent to a provider that
trains on its inputs. Encoding that as a hard failure — rather than as a
convention someone has to remember — is the whole point.

Design notes:
    * Clearance fails **closed**. An unknown provider is treated as having no
      clearance, and an unknown data class is treated as maximally sensitive.
    * ``SECRET`` is cleared by nothing. Credentials are never sent to a model,
      not even a local one; if a caller asks, that is a bug worth raising.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final


class DataClass(StrEnum):
    """How sensitive a payload is.

    Ordered by increasing sensitivity. Callers should classify *minimally
    permissively*: when in doubt, choose the more sensitive class.
    """

    #: Synthetic or public data. The in-repo demo app, generated fixtures,
    #: published documentation. Safe for free tiers that train on inputs.
    PUBLIC = "public"

    #: Real source code, real application state, internal documents, customer
    #: or employee data. Must stay on providers with an explicit no-training
    #: guarantee, or local inference.
    PROPRIETARY = "proprietary"

    #: Credentials, tokens, keys, session cookies. Cleared by nothing. Should
    #: be redacted before it ever gets this far; this class exists so that a
    #: redaction failure is loud instead of silent.
    SECRET = "secret"

    @property
    def rank(self) -> int:
        return _RANK[self]


class ProviderName(StrEnum):
    """Known model providers."""

    GEMINI = "gemini"
    NVIDIA = "nvidia"
    OLLAMA = "ollama"


_RANK: Final[dict[DataClass, int]] = {
    DataClass.PUBLIC: 0,
    DataClass.PROPRIETARY: 1,
    DataClass.SECRET: 2,
}

#: Data classes each provider is permitted to receive.
#:
#: GEMINI is tier 1 and free. Google's free tier may use prompts to improve its
#: products, with human review, so it is restricted to PUBLIC payloads. This is
#: why the white-box triage path (which reads real source) can never fall back
#: to tier 1 — ``require_clearance`` will raise instead.
#:
#: NVIDIA hosts tier 2. Its catalogue is used here for prototyping; the
#: training policy was NOT independently verified, so PROPRIETARY access is a
#: deliberate, documented assumption that must be re-checked before this tool
#: points at a real repository. Tracked in ARCHITECTURE.md.
#:
#: OLLAMA is local, so nothing leaves the machine and it can hold PROPRIETARY.
#: It still never sees SECRET: a credential has no business in a prompt.
CLEARANCES: Final[dict[ProviderName, frozenset[DataClass]]] = {
    ProviderName.GEMINI: frozenset({DataClass.PUBLIC}),
    ProviderName.NVIDIA: frozenset({DataClass.PUBLIC, DataClass.PROPRIETARY}),
    ProviderName.OLLAMA: frozenset({DataClass.PUBLIC, DataClass.PROPRIETARY}),
}


class SensitivityViolation(PermissionError):
    """Raised when a payload is not cleared to reach the selected provider.

    Deliberately a :class:`PermissionError` so that a broad ``except Exception``
    higher up cannot silently swallow a data-handling breach.
    """

    def __init__(self, provider: ProviderName, data_class: DataClass, detail: str = "") -> None:
        self.provider = provider
        self.data_class = data_class
        message = (
            f"Refusing to send {data_class.value!r} data to provider "
            f"{provider.value!r}, which has no clearance for it."
        )
        if detail:
            message = f"{message} {detail}"
        super().__init__(message)


def is_cleared(provider: ProviderName, data_class: DataClass) -> bool:
    """Return whether ``provider`` may receive ``data_class`` payloads."""
    return data_class in CLEARANCES.get(provider, frozenset())


def require_clearance(provider: ProviderName, data_class: DataClass) -> None:
    """Raise :class:`SensitivityViolation` unless the pairing is permitted.

    Call this immediately before constructing a client or issuing a request.
    It must not be skipped for "read-only" or "small" payloads: partial source
    code is still source code.
    """
    if data_class is DataClass.SECRET:
        raise SensitivityViolation(
            provider,
            data_class,
            "Credentials must be redacted before reaching the model layer.",
        )
    if not is_cleared(provider, data_class):
        raise SensitivityViolation(provider, data_class)


def cleared_providers(data_class: DataClass) -> frozenset[ProviderName]:
    """Return every provider permitted to receive ``data_class``."""
    return frozenset(
        provider for provider in ProviderName if is_cleared(provider, data_class)
    )


def classify_path(path: str) -> DataClass:
    """Best-effort classification of a filesystem path.

    Anything under the in-repo demo application counts as synthetic, and is
    therefore safe for tier 1. Everything else is treated as proprietary,
    because a path we do not recognise is far more likely to be real code than
    a fixture. Callers may always override with an explicit :class:`DataClass`.
    """
    normalized = path.replace("\\", "/").lower()
    # Markers deliberately omit a leading slash so they match both relative and
    # absolute paths without needing to normalise the prefix.
    synthetic_markers = ("apps/guinea_pig/", "tests/fixtures/")
    if any(marker in normalized for marker in synthetic_markers):
        return DataClass.PUBLIC
    return DataClass.PROPRIETARY
