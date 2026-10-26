"""Tests for zero_jax.scipy.stats.multivariate_normal."""

from __future__ import annotations

import math

import zero_jax.numpy as jnp
import zero_jax.scipy.stats.multivariate_normal as mod
from tests.conftest import assert_allclose


def test_logpdf() -> None:
    """Test logpdf."""
    assert_allclose(mod.logpdf(jnp.array([0.0, 0.0]), jnp.array([0.0, 0.0]), jnp.eye(2)), -math.log(2.0 * math.pi), atol=1e-5)


def test_pdf() -> None:
    """Test pdf."""
    assert_allclose(mod.pdf(jnp.array([0.0, 0.0]), jnp.array([0.0, 0.0]), jnp.eye(2)), 1.0 / (2.0 * math.pi), atol=1e-5)
