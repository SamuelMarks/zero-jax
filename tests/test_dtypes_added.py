"""Tests for tests/test_dtypes_added.py."""

from __future__ import annotations

import zero_jax.dtypes as mod


def test_float0() -> None:
    """Verify float0 returns the symbolic tangent dtype identifier."""
    assert mod.float0() == "float0"
