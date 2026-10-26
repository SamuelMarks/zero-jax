"""Tests for zero_jax.custom_transpose."""

from __future__ import annotations

import zero_jax.custom_transpose as mod


def test_custom_transpose() -> None:
    """Test custom_transpose."""
    empty = mod.custom_transpose()
    assert empty() is None
    ct = mod.custom_transpose(lambda x: x * 3)
    rule = lambda g: (g * 3,)
    assert ct.def_transpose(rule) is rule
    assert ct(4) == 12
