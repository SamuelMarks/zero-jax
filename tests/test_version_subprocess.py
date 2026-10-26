"""Tests for zero_jax.version.subprocess module."""

from __future__ import annotations

import zero_jax.version.subprocess as mod


def test_module_exists() -> None:
    """Verify zero_jax.version.subprocess exports CalledProcessError and check_output."""
    err = mod.CalledProcessError(1, "cmd")
    assert err.returncode == 1
    assert callable(mod.check_output)
