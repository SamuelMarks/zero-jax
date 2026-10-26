"""Tests for zero_jax.version.pathlib module."""

from __future__ import annotations

import zero_jax.version.pathlib as mod


def test_module_exists() -> None:
    """Verify zero_jax.version.pathlib exports Path."""
    p = mod.Path("/tmp/example.txt")
    assert p.name == "example.txt"
