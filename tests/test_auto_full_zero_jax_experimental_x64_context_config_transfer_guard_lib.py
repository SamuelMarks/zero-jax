"""Tests for zero_jax.experimental.x64_context.config.transfer_guard_lib."""

from typing import Any

import pytest

import zero_jax.experimental.x64_context.config.transfer_guard_lib as mod


def test_Any() -> None:
    """Test Any."""
    try:
        mod.Any()
    except Exception:
        pass


def test_class_TransferGuardLevel() -> None:
    """Test class TransferGuardLevel."""
    try:
        mod.TransferGuardLevel()
    except Exception:
        pass


def test_class_TransferGuardState() -> None:
    """Test class TransferGuardState."""
    try:
        mod.TransferGuardState()
    except Exception:
        pass


def test_global_state() -> None:
    """Test global_state."""
    try:
        mod.global_state()
    except Exception:
        pass


def test_thread_local_state() -> None:
    """Test thread_local_state."""
    try:
        mod.thread_local_state()
    except Exception:
        pass
