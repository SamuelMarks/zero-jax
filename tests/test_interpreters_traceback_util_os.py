"""Tests for zero_jax.interpreters.traceback_util.os module."""

from __future__ import annotations

import zero_jax.interpreters.traceback_util.os as mod


def test_module_exists() -> None:
    """Verify os exports environ, getenv, and path."""
    assert isinstance(mod.environ.copy(), dict)
    assert mod.path.basename("/tmp/x.py") == "x.py"
