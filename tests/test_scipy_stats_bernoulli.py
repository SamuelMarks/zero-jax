"""Tests for zero_jax.scipy.stats.bernoulli."""

from __future__ import annotations

import zero_jax.numpy as jnp
import zero_jax.scipy.stats.bernoulli as mod
from tests.conftest import assert_allclose


def test_cdf() -> None:
    """Test cdf."""
    assert_allclose(mod.cdf(jnp.array([-1.0, 0.0, 1.0]), 0.3), [0.0, 0.7, 1.0], atol=1e-5)


def test_logpmf() -> None:
    """Test logpmf."""
    assert_allclose(mod.logpmf(jnp.array([0.0, 1.0]), 0.25), [jnp.log(0.75), jnp.log(0.25)], atol=1e-5)


def test_pmf() -> None:
    """Test pmf."""
    assert_allclose(mod.pmf(jnp.array([0.0, 1.0]), 0.25), [0.75, 0.25], atol=1e-5)


def test_ppf() -> None:
    """Test ppf."""
    assert_allclose(mod.ppf(jnp.array([0.2, 0.9]), 0.5), [0.0, 1.0], atol=1e-5)
