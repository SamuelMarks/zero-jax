"""Tests for zero_jax.scipy."""

from __future__ import annotations

import zero_jax.scipy as mod


def test_module_exists() -> None:
    """Test scipy submodules are exposed."""
    assert "linalg" in mod.__all__
    assert "special" in mod.__all__
    assert "stats" in mod.__all__
