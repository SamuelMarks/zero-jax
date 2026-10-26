"""Tests for zero_jax.interpreters.traceback_util.types module."""

from __future__ import annotations

import zero_jax.interpreters.traceback_util.types as mod


def test_module_exists() -> None:
    """Verify types exports ModuleType and TracebackType."""
    assert isinstance(mod, mod.ModuleType)
    assert isinstance(mod.TracebackType, type)
