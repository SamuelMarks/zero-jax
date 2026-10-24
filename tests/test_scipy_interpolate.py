"""Tests for zero_jax module."""

from __future__ import annotations

import zero_jax.scipy.interpolate as mod


def test_RegularGridInterpolator() -> None:
    """Test RegularGridInterpolator."""
    obj = mod.RegularGridInterpolator()
    assert obj is not None
