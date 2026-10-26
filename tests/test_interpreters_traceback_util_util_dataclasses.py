"""Tests for zero_jax.interpreters.traceback_util.util.dataclasses module."""

from __future__ import annotations

import zero_jax.interpreters.traceback_util.util.dataclasses as mod


def test_module_exists() -> None:
    """Verify dataclasses exports dataclass and is_dataclass."""
    assert callable(mod.dataclass)
    assert mod.is_dataclass(int) is False
