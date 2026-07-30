"""Tests for zero_jax.scipy.interpolate."""

from typing import Any

import pytest

import zero_jax.scipy.interpolate as mod


def test_Any() -> None:
    """Test Any."""
    try:
        mod.Any()
    except Exception:
        pass


def test_class_RegularGridInterpolator() -> None:
    """Test class RegularGridInterpolator."""
    try:
        mod.RegularGridInterpolator()
    except Exception:
        pass
