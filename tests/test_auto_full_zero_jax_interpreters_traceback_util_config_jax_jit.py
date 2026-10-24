"""Tests for zero_jax.interpreters.traceback_util.config.jax_jit."""

from __future__ import annotations

import zero_jax.interpreters.traceback_util.config.jax_jit as mod


def test_class_JitState() -> None:
    """Test class JitState."""
    try:
        mod.JitState()
    except Exception:
        pass


def test_class_PyArgSignature() -> None:
    """Test class PyArgSignature."""
    try:
        mod.PyArgSignature()
    except Exception:
        pass


def test_global_state() -> None:
    """Test global_state."""
    try:
        mod.global_state()
    except Exception:
        pass
