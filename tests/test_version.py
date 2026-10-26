"""Tests for zero_jax.version module."""

from __future__ import annotations

import zero_jax.version as mod


def test_module_exists() -> None:
    """Verify zero_jax.version exposes version metadata and annotations helper."""
    assert mod.__version__ == "0.0.5"
    assert mod.__version_info__ == (0, 0, 5)
    assert mod.annotations() == "__future__.annotations"
    assert mod.annotations("custom") == "custom"
