"""Tests for zero_jax.scipy.stats.geom."""

from __future__ import annotations

import zero_jax.numpy as jnp
import zero_jax.scipy.stats.geom as mod
from tests.conftest import assert_allclose


def test_logpmf() -> None:
    """Test logpmf."""
    assert_allclose(mod.logpmf(jnp.array([1.0, 2.0]), 0.5), [jnp.log(0.5), jnp.log(0.25)], atol=1e-5)


def test_pmf() -> None:
    """Test pmf."""
    assert_allclose(mod.pmf(jnp.array([1.0, 2.0]), 0.5), [0.5, 0.25], atol=1e-5)
