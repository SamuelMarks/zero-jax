"""Tests for zero_jax.interpreters.traceback_util.util.config.logging_config."""

from __future__ import annotations

import zero_jax.interpreters.traceback_util.util.config.logging_config as mod


def test_logging_config() -> None:
    """Verify enable_debug_logging and disable_all_debug_logging."""
    assert mod.enable_debug_logging("jax") == "jax"
    assert mod.disable_all_debug_logging() is False
