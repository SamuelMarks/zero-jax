"""Tests for zero_jax.scipy.cluster."""

from __future__ import annotations

import zero_jax.scipy.cluster as mod


def test_module_exists() -> None:
    """Test scipy.cluster exposes vq."""
    assert "vq" in mod.__all__
