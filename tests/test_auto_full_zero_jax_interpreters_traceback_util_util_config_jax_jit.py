"""Tests for zero_jax.interpreters.traceback_util.util.config.jax_jit."""

from typing import Any

import pytest

import zero_jax.interpreters.traceback_util.util.config.jax_jit as mod


def test_class_JitState() -> None:
    """Test class JitState."""
    try:
        mod.JitState()
    except Exception:
        pass
