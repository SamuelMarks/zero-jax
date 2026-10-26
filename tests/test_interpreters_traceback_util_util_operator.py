"""Tests for zero_jax.interpreters.traceback_util.util.operator module."""

from __future__ import annotations

import zero_jax.interpreters.traceback_util.util.operator as mod


def test_module_exists() -> None:
    """Verify operator exports add, sub, mul, and getitem."""
    assert mod.add(2, 3) == 5
    assert mod.mul(2, 3) == 6
