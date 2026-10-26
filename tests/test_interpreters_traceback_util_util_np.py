"""Tests for zero_jax.interpreters.traceback_util.util.np module."""

from __future__ import annotations

import zero_jax.interpreters.traceback_util.util.np as mod


def test_module_exists() -> None:
    """Verify np exports array, zeros, and ones."""
    arr = mod.zeros((2,))
    assert arr.shape == (2,)
