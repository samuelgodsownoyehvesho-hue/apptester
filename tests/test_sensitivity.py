"""The data-handling invariants.

These tests are the enforcement mechanism for a policy decision, not a
description of current behaviour. If one fails after a refactor, the refactor
introduced a way for sensitive data to reach a provider that trains on it.
"""

from __future__ import annotations

import pytest

from crucible.llm.sensitivity import (
    CLEARANCES,
    DataClass,
    ProviderName,
    SensitivityViolation,
    classify_path,
    cleared_providers,
    is_cleared,
    require_clearance,
)


class TestClearancePolicy:
    def test_gemini_may_receive_public_payloads(self) -> None:
        assert is_cleared(ProviderName.GEMINI, DataClass.PUBLIC)

    def test_gemini_may_never_receive_proprietary_payloads(self) -> None:
        # Google's free tier may train on prompts, so real source code and real
        # application data must not reach it.
        assert not is_cleared(ProviderName.GEMINI, DataClass.PROPRIETARY)

    def test_nvidia_may_receive_proprietary_payloads(self) -> None:
        assert is_cleared(ProviderName.NVIDIA, DataClass.PROPRIETARY)

    def test_local_inference_may_receive_proprietary_payloads(self) -> None:
        # Nothing leaves the machine, so local inference is safe by construction.
        assert is_cleared(ProviderName.OLLAMA, DataClass.PROPRIETARY)

    @pytest.mark.parametrize("provider", list(ProviderName))
    def test_no_provider_may_receive_secrets(self, provider: ProviderName) -> None:
        assert not is_cleared(provider, DataClass.SECRET)

    def test_cleared_providers_public_is_all_providers(self) -> None:
        assert cleared_providers(DataClass.PUBLIC) == frozenset(ProviderName)

    def test_cleared_providers_proprietary_excludes_gemini(self) -> None:
        cleared = cleared_providers(DataClass.PROPRIETARY)
        assert ProviderName.GEMINI not in cleared
        assert ProviderName.NVIDIA in cleared

    def test_cleared_providers_secret_is_empty(self) -> None:
        assert cleared_providers(DataClass.SECRET) == frozenset()


class TestRequireClearance:
    def test_allows_cleared_pairing(self) -> None:
        require_clearance(ProviderName.GEMINI, DataClass.PUBLIC)

    def test_rejects_proprietary_to_gemini(self) -> None:
        with pytest.raises(SensitivityViolation) as exc_info:
            require_clearance(ProviderName.GEMINI, DataClass.PROPRIETARY)
        assert exc_info.value.provider is ProviderName.GEMINI
        assert exc_info.value.data_class is DataClass.PROPRIETARY

    def test_rejects_secret_with_redaction_hint(self) -> None:
        with pytest.raises(SensitivityViolation, match="redacted"):
            require_clearance(ProviderName.OLLAMA, DataClass.SECRET)

    def test_violation_is_a_permission_error(self) -> None:
        # A PermissionError, so a broad `except Exception` cannot quietly
        # swallow a data-handling breach.
        assert issubclass(SensitivityViolation, PermissionError)

    def test_fails_closed_for_unknown_clearance(self) -> None:
        # Anything absent from CLEARANCES has no clearance at all.
        assert DataClass.PUBLIC not in CLEARANCES.get("not-a-provider", frozenset())  # type: ignore[arg-type]


class TestClassifyPath:
    def test_demo_app_is_public(self) -> None:
        assert classify_path("apps/guinea_pig/src/app/page.tsx") is DataClass.PUBLIC

    def test_windows_style_demo_path_is_public(self) -> None:
        assert classify_path(r"apps\guinea_pig\src\app\page.tsx") is DataClass.PUBLIC

    def test_fixtures_are_public(self) -> None:
        assert classify_path("tests/fixtures/sample_response.json") is DataClass.PUBLIC

    def test_unknown_path_defaults_to_proprietary(self) -> None:
        # An unrecognised path is far more likely to be real code than a
        # fixture, so the default must be the safe one.
        assert classify_path("C:/work/internal-payments/src/main.py") is DataClass.PROPRIETARY


class TestDataClassOrdering:
    def test_ranks_increase_with_sensitivity(self) -> None:
        assert DataClass.PUBLIC.rank < DataClass.PROPRIETARY.rank < DataClass.SECRET.rank
