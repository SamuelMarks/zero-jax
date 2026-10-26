"""Tests for zero_jax.scipy.stats.binom."""

from __future__ import annotations

import zero_jax.numpy as jnp
import zero_jax.scipy.stats.binom as mod
from tests.conftest import assert_allclose


def test_logpmf() -> None:
    """Test logpmf."""
    res = mod.logpmf(jnp.array([0.0, 1.0, 2.0]), 2.0, 0.5)
    assert_allclose(jnp.exp(res), [0.25, 0.5, 0.25], atol=1e-5)


def test_pmf() -> None:
    """Test pmf."""
    assert_allclose(mod.pmf(jnp.array([0.0, 1.0, 2.0]), 2.0, 0.5), [0.25, 0.5, 0.25], atol=1e-5)
