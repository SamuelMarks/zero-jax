"""Tests for zero_jax.interpreters.traceback_util.functools module."""

from __future__ import annotations

import zero_jax.interpreters.traceback_util.functools as mod


def test_module_exists() -> None:
    """Verify functools exports partial, lru_cache, and wraps."""
    add_one = mod.partial(lambda a, b: a + b, 1)
    assert add_one(2) == 3
    assert callable(mod.lru_cache)
    assert callable(mod.wraps)
