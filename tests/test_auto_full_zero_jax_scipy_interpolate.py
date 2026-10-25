"""Tests for zero_jax.scipy.interpolate."""

from __future__ import annotations

import zero_jax.numpy as jnp
import zero_jax.scipy.interpolate as mod
from tests.conftest import assert_allclose


def test_class_RegularGridInterpolator() -> None:
    """Test class RegularGridInterpolator."""
    grid = (jnp.array([0.0, 1.0, 2.0]),)
    values = jnp.array([0.0, 10.0, 20.0])
    interp = mod.RegularGridInterpolator(grid, values)
    res = interp(jnp.array([0.5, 1.5]))
    assert_allclose(res, [5.0, 15.0])

    empty_interp = mod.RegularGridInterpolator()
    assert empty_interp(jnp.array([1.0])).shape == (1,)
