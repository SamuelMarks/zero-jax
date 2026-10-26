"""Tests for zero_jax.interpreters.traceback_util.util.it module."""

from __future__ import annotations

import zero_jax.interpreters.traceback_util.util.it as mod


def test_module_exists() -> None:
    """Verify it exports chain, count, and islice."""
    assert list(mod.chain([1], [2])) == [1, 2]
    assert list(mod.islice(mod.count(0), 3)) == [0, 1, 2]
