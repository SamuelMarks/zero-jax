"""Tests for zero_jax.scipy.interpolate."""

from __future__ import annotations

import math

import pytest

import zero_jax.numpy as jnp
import zero_jax.scipy.interpolate as mod
from tests.conftest import assert_allclose


def test_RegularGridInterpolator() -> None:
    """Test RegularGridInterpolator in 1D and 2D with linear, nearest, bounds_error, and fill_value."""
    empty_obj = mod.RegularGridInterpolator()
    assert_allclose(empty_obj(jnp.array([1.0, 2.0])), [1.0, 2.0])

    grid1d = (jnp.array([0.0, 1.0, 2.0]),)
    vals1d = jnp.array([0.0, 10.0, 20.0])
    interp_near = mod.RegularGridInterpolator(grid1d, vals1d, method="nearest")
    assert_allclose(interp_near(jnp.array([0.2, 1.8])), [0.0, 20.0])
    assert_allclose(interp_near(jnp.array([[0.5], [1.5]]), method="linear"), [5.0, 15.0])

    # Out-of-bounds fill_value (default NaN vs explicit float vs None)
    oob_res = interp_near(jnp.array([-1.0, 1.0]))
    assert math.isnan(float(oob_res[0].item()))
    assert_allclose(oob_res[1], 10.0)

    interp_extrap = mod.RegularGridInterpolator(grid1d, vals1d, fill_value=None)
    assert_allclose(interp_extrap(jnp.array([-1.0, 3.0])), [0.0, 20.0])

    interp_err = mod.RegularGridInterpolator(grid1d, vals1d, bounds_error=True)
    with pytest.raises(ValueError, match="out of bounds"):
        interp_err(jnp.array([-0.5]))

    with pytest.raises(ValueError, match="not defined"):
        interp_near(jnp.array([0.5]), method="cubic")

    # 2D linear and nearest interpolation
    gx = jnp.array([0.0, 1.0])
    gy = jnp.array([0.0, 1.0])
    vals2d = jnp.array([[0.0, 2.0], [4.0, 6.0]])
    interp2d = mod.RegularGridInterpolator((gx, gy), vals2d, method="linear")
    assert_allclose(interp2d(jnp.array([[0.5, 0.5], [1.0, 0.5]])), [3.0, 5.0])
    assert_allclose(interp2d(jnp.array([[0.1, 0.9]]), method="nearest"), [2.0])
