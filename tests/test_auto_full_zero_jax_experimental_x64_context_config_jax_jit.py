"""Tests for zero_jax.experimental.x64_context.config.jax_jit."""

from __future__ import annotations

import zero_jax.experimental.x64_context.config.jax_jit as mod


def test_class_JitState() -> None:
    """Test class JitState."""
    obj = mod.JitState(disable_jit=True)
    assert isinstance(obj, mod.JitState)
    assert obj.disable_jit is True


def test_class_PyArgSignature() -> None:
    """Test class PyArgSignature."""
    obj = mod.PyArgSignature("float32", (2, 3), False)
    assert isinstance(obj, mod.PyArgSignature)
    assert obj.shape == (2, 3)


def test_get_enable_x64() -> None:
    """Test get_enable_x64."""
    assert isinstance(mod.get_enable_x64(), bool)


def test_global_state() -> None:
    """Test global_state."""
    state = mod.global_state()
    assert isinstance(state, mod.JitState)


def test_set_thread_local_state_initialization_callback() -> None:
    """Test set_thread_local_state_initialization_callback."""
    cb = lambda: None
    assert mod.set_thread_local_state_initialization_callback(cb) is cb


def test_swap_thread_local_state_disable_jit() -> None:
    """Test swap_thread_local_state_disable_jit."""
    prev = mod.swap_thread_local_state_disable_jit(True)
    assert isinstance(prev, bool)
    assert mod.swap_thread_local_state_disable_jit(prev) is True


def test_thread_local_state() -> None:
    """Test thread_local_state."""
    state = mod.thread_local_state()
    assert isinstance(state, mod.JitState)
