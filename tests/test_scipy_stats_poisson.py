"""Tests for zero_jax.scipy.stats.poisson."""

from __future__ import annotations

import zero_jax.numpy as jnp
import zero_jax.scipy.stats.poisson as mod
from tests.conftest import assert_allclose


def test_cdf() -> None:
    """Test cdf."""
    assert_allclose(mod.cdf(jnp.array([-1.0, 0.0]), 1.0), [0.0, jnp.exp(-1.0)], atol=1e-4)


def test_logpmf() -> None:
    """Test logpmf."""
    assert_allclose(mod.logpmf(jnp.array([0.0, 1.0]), 1.0), [-1.0, -1.0], atol=1e-4)


def test_pmf() -> None:
    """Test pmf."""
    assert_allclose(mod.pmf(jnp.array([0.0, 1.0]), 1.0), [jnp.exp(-1.0), jnp.exp(-1.0)], atol=1e-4)
