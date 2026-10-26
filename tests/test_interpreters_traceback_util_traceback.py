"""Tests for zero_jax.interpreters.traceback_util.traceback module."""

from __future__ import annotations

import zero_jax.interpreters.traceback_util.traceback as mod


def test_module_exists() -> None:
    """Verify traceback exports format_exception_only."""
    lines = mod.format_exception_only(ValueError, ValueError("err"))
    assert len(lines) >= 1
