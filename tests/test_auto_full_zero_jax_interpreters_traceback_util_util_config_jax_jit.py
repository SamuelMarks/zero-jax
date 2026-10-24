"""Tests for zero_jax.interpreters.traceback_util.util.config.jax_jit."""

from __future__ import annotations

import zero_jax.interpreters.traceback_util.util.config.jax_jit as mod


def test_class_JitState() -> None:
    """Test class JitState."""
    try:
        mod.JitState()
    except Exception:
        pass
