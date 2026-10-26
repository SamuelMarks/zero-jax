"""Tests for zero_jax.interpreters.os module."""

from __future__ import annotations

import zero_jax.interpreters.os as mod


def test_module_exists() -> None:
    """Verify zero_jax.interpreters.os exports environ, getenv, and path."""
    assert isinstance(mod.environ.copy(), dict)
    assert callable(mod.getenv)
    assert mod.path.basename("/tmp/a.py") == "a.py"
