"""Tests for zero_jax.interpreters.traceback_util.util.logging module."""

from __future__ import annotations

import zero_jax.interpreters.traceback_util.util.logging as mod


def test_module_exists() -> None:
    """Verify logging exports getLogger and Logger."""
    assert isinstance(mod.getLogger("jax"), mod.Logger)
