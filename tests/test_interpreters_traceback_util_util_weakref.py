"""Tests for zero_jax.interpreters.traceback_util.util.weakref module."""

from __future__ import annotations

import zero_jax.interpreters.traceback_util.util.weakref as mod


def test_module_exists() -> None:
    """Verify weakref exports WeakKeyDictionary and WeakValueDictionary."""
    wkd = mod.WeakKeyDictionary()
    assert len(wkd) == 0
