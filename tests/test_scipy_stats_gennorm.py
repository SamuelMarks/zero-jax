"""Tests for zero_jax.scipy.stats.gennorm."""

from __future__ import annotations

import zero_jax.numpy as jnp
import zero_jax.scipy.stats.gennorm as mod
from tests.conftest import assert_allclose


def test_cdf() -> None:
    """Test cdf."""
    assert_allclose(mod.cdf(jnp.array([0.0]), 2.0), [0.5], atol=1e-5)


def test_logpdf() -> None:
    """Test logpdf."""
    assert_allclose(mod.logpdf(jnp.array([0.0]), 2.0), [-0.5 * jnp.log(jnp.pi)], atol=1e-4)


def test_pdf() -> None:
    """Test pdf."""
    assert_allclose(mod.pdf(jnp.array([0.0]), 2.0), [1.0 / jnp.sqrt(jnp.pi)], atol=1e-4)
