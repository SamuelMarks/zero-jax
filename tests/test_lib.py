"""Tests for zero_jax.lib."""

from __future__ import annotations

import zero_jax.lib as mod


def test_module_exists() -> None:
    """Test zero_jax.lib exposes xla_bridge, xla_client, and xla_extension."""
    assert "xla_bridge" in mod.__all__
    assert "xla_client" in mod.__all__
    assert "xla_extension" in mod.__all__
