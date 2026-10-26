"""Tests for zero_jax.interpreters.traceback_util.util.functools module."""

from __future__ import annotations

import zero_jax.interpreters.traceback_util.util.functools as mod


def test_module_exists() -> None:
    """Verify functools exports partial and reduce."""
    assert mod.reduce(lambda a, b: a + b, [1, 2, 3]) == 6
    assert callable(mod.partial)
