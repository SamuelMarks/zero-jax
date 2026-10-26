"""Tests for zero_jax.interpreters.traceback_util.util.abc module."""

from __future__ import annotations

import zero_jax.interpreters.traceback_util.util.abc as mod


def test_module_exists() -> None:
    """Verify abc exports ABC and ABCMeta."""
    assert isinstance(mod.ABC, mod.ABCMeta)
    assert callable(mod.abstractmethod)
