"""Tests for zero_jax.scipy.ndimage."""

from __future__ import annotations

import zero_jax.numpy as jnp
import zero_jax.scipy.ndimage as mod
from tests.conftest import assert_allclose


def test_map_coordinates() -> None:
    """Test map_coordinates."""
    inp = jnp.array([0.0, 10.0, 20.0])
    out1 = mod.map_coordinates(inp, [jnp.array([0.5, 1.5])], 1)
    assert_allclose(out1, [5.0, 15.0])
    out0 = mod.map_coordinates(inp, [jnp.array([0.2, 1.8])], 0)
    assert_allclose(out0, [0.0, 20.0])
