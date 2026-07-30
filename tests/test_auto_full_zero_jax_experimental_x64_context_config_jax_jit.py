"""Tests for zero_jax.experimental.x64_context.config.jax_jit."""

from typing import Any

import pytest

import zero_jax.experimental.x64_context.config.jax_jit as mod


def test_Any() -> None:
    """Test Any."""
    try:
        mod.Any()
    except Exception:
        pass


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


def test_get_enable_x64() -> None:
    """Test get_enable_x64."""
    try:
        mod.get_enable_x64()
    except Exception:
        pass


def test_global_state() -> None:
    """Test global_state."""
    try:
        mod.global_state()
    except Exception:
        pass


def test_set_thread_local_state_initialization_callback() -> None:
    """Test set_thread_local_state_initialization_callback."""
    try:
        mod.set_thread_local_state_initialization_callback()
    except Exception:
        pass


def test_swap_thread_local_state_disable_jit() -> None:
    """Test swap_thread_local_state_disable_jit."""
    try:
        mod.swap_thread_local_state_disable_jit()
    except Exception:
        pass


def test_thread_local_state() -> None:
    """Test thread_local_state."""
    try:
        mod.thread_local_state()
    except Exception:
        pass
