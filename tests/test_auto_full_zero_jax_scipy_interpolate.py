"""Tests for zero_jax.scipy.interpolate."""

from __future__ import annotations

import zero_jax.scipy.interpolate as mod


def test_class_RegularGridInterpolator() -> None:
    """Test class RegularGridInterpolator."""
    try:
        mod.RegularGridInterpolator()
    except Exception:
        pass
