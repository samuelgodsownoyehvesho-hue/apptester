"""Shared test fixtures.

Every settings object is built explicitly with ``_env_file=None`` so tests
never read the developer's real ``.env`` and never depend on ambient keys.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from crucible.core.config import Settings


@pytest.fixture
def settings_factory() -> Callable[..., Settings]:
    """Return a factory building a Settings object with no providers enabled."""

    def _make(**overrides: Any) -> Settings:
        base: dict[str, Any] = {
            "_env_file": None,
            "gemini_api_key": "",
            "nvidia_api_key": "",
            "enable_gemini": False,
            "enable_nvidia": False,
            "enable_ollama": False,
        }
        base.update(overrides)
        return Settings(**base)

    return _make


@pytest.fixture
def gemini_only() -> Settings:
    return Settings(
        _env_file=None,
        gemini_api_key="test-gemini-key",
        nvidia_api_key="",
        enable_gemini=True,
        enable_nvidia=False,
        enable_ollama=False,
    )


@pytest.fixture
def gemini_and_nvidia() -> Settings:
    return Settings(
        _env_file=None,
        gemini_api_key="test-gemini-key",
        nvidia_api_key="test-nvidia-key",
        enable_gemini=True,
        enable_nvidia=True,
        enable_ollama=False,
    )


@pytest.fixture
def nvidia_only() -> Settings:
    return Settings(
        _env_file=None,
        gemini_api_key="",
        nvidia_api_key="test-nvidia-key",
        enable_gemini=False,
        enable_nvidia=True,
        enable_ollama=False,
    )
