"""Tests for zero_jax.interpreters.traceback_util.config.jax_jit."""

from __future__ import annotations

import zero_jax.interpreters.traceback_util.config.jax_jit as mod


def test_class_JitState() -> None:
    """Test class JitState."""
    obj = mod.JitState()
    assert isinstance(obj, mod.JitState)
    assert obj.disable_jit is False


def test_class_PyArgSignature() -> None:
    """Test class PyArgSignature."""
    obj = mod.PyArgSignature()
    assert isinstance(obj, mod.PyArgSignature)
    assert obj.dtype == "float32"


def test_global_state() -> None:
    """Test global_state."""
    assert isinstance(mod.global_state(), mod.JitState)
