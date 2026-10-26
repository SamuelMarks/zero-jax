"""Tests for zero_jax.distributed."""

from __future__ import annotations

import zero_jax.distributed as mod


def test_initialize() -> None:
    """Test initialize."""
    mod.initialize(coordinator_address="127.0.0.1:1234", num_processes=1, process_id=0)
    assert mod.is_initialized() is True


def test_shutdown() -> None:
    """Test shutdown."""
    mod.initialize()
    assert mod.is_initialized() is True
    mod.shutdown()
    assert mod.is_initialized() is False
