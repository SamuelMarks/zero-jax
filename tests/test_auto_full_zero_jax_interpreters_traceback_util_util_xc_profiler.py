"""Tests for zero_jax.interpreters.traceback_util.util.xc.profiler."""

from __future__ import annotations

import zero_jax.interpreters.traceback_util.util.xc.profiler as mod


def test_class_TraceMe() -> None:
    """Test class TraceMe."""
    obj = mod.TraceMe()
    assert isinstance(obj, mod.TraceMe)
    assert obj.name == "trace_me"
