"""Tests for zero_jax.version.os module."""

from __future__ import annotations

import zero_jax.version.os as mod


def test_module_exists() -> None:
    """Verify zero_jax.version.os exports environ and path."""
    assert isinstance(mod.environ.copy(), dict)
    assert mod.path.basename("/tmp/example.txt") == "example.txt"
