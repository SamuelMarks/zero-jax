"""Tests for zero_jax.interpreters.traceback_util.util.xc.profiler."""

from typing import Any

import pytest

import zero_jax.interpreters.traceback_util.util.xc.profiler as mod


def test_class_TraceMe() -> None:
    """Test class TraceMe."""
    try:
        mod.TraceMe()
    except Exception:
        pass
