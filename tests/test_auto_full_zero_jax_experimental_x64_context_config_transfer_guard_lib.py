"""Tests for zero_jax.experimental.x64_context.config.transfer_guard_lib."""

from __future__ import annotations

import zero_jax.experimental.x64_context.config.transfer_guard_lib as mod


def test_class_TransferGuardLevel() -> None:
    """Test class TransferGuardLevel."""
    obj = mod.TransferGuardLevel("disallow")
    assert isinstance(obj, mod.TransferGuardLevel)
    assert obj.level == "disallow"


def test_class_TransferGuardState() -> None:
    """Test class TransferGuardState."""
    obj = mod.TransferGuardState(host_to_device="log")
    assert isinstance(obj, mod.TransferGuardState)
    assert obj.host_to_device == "log"


def test_global_state() -> None:
    """Test global_state."""
    state = mod.global_state()
    assert isinstance(state, mod.TransferGuardState)


def test_thread_local_state() -> None:
    """Test thread_local_state."""
    state = mod.thread_local_state()
    assert isinstance(state, mod.TransferGuardState)
