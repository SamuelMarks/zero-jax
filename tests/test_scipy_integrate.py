"""Tests for zero_jax.scipy.integrate."""

from __future__ import annotations

import zero_jax.numpy as jnp
import zero_jax.scipy.integrate as mod
from tests.conftest import assert_allclose


def test_trapezoid() -> None:
    """Test trapezoid."""
    assert_allclose(mod.trapezoid(jnp.array([1.0, 2.0, 3.0])), 4.0, atol=1e-5)
    assert_allclose(mod.trapezoid(jnp.array([1.0, 2.0, 3.0]), x=jnp.array([0.0, 2.0, 4.0])), 8.0, atol=1e-5)
