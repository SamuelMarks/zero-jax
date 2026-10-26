"""Tests for zero_jax.lib.xla_bridge.xla_client."""

from __future__ import annotations

import zero_jax.lib.xla_bridge.xla_client as mod


def test_module_exists() -> None:
    """Test xla_bridge.xla_client exposes __all__."""
    assert mod.__all__ == []
