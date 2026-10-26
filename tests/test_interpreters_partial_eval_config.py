"""Tests for zero_jax.interpreters.partial_eval.config module."""

from __future__ import annotations

import zero_jax.interpreters.partial_eval.config as mod


def test_module_exists() -> None:
    """Verify zero_jax.interpreters.partial_eval.config exports Config and jax_jit."""
    cfg = mod.Config()
    assert isinstance(cfg, mod.Config)
    assert mod.jax_jit is True
