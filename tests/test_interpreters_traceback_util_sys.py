"""Tests for zero_jax.interpreters.traceback_util.sys module."""

from __future__ import annotations

import zero_jax.interpreters.traceback_util.sys as mod


def test_module_exists() -> None:
    """Verify sys exports exc_info and modules."""
    assert isinstance(mod.exc_info(), tuple)
    assert isinstance(mod.modules, dict)
