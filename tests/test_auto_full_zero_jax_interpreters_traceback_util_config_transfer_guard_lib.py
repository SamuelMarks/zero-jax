"""Tests for zero_jax.interpreters.traceback_util.config.transfer_guard_lib."""

from __future__ import annotations

import zero_jax.interpreters.traceback_util.config.transfer_guard_lib as mod


def test_transfer_guard_lib() -> None:
    """Verify TransferGuardLevel, TransferGuardState, global_state, and thread_local_state."""
    lvl = mod.TransferGuardLevel("allow")
    assert lvl.level == "allow"
    assert isinstance(mod.global_state(), mod.TransferGuardState)
    assert isinstance(mod.thread_local_state(), mod.TransferGuardState)
