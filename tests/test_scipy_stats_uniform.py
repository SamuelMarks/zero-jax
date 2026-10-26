"""Tests for zero_jax.scipy.stats.uniform."""

from __future__ import annotations

import zero_jax.numpy as jnp
import zero_jax.scipy.stats.uniform as mod
from tests.conftest import assert_allclose


def test_cdf() -> None:
    """Test cdf."""
    assert_allclose(mod.cdf(jnp.array([-0.5, 0.5, 1.5])), [0.0, 0.5, 1.0], atol=1e-5)


def test_logpdf() -> None:
    """Test logpdf."""
    assert_allclose(mod.logpdf(jnp.array([0.5])), [0.0], atol=1e-5)


def test_pdf() -> None:
    """Test pdf."""
    assert_allclose(mod.pdf(jnp.array([-0.5, 0.5, 1.5])), [0.0, 1.0, 0.0], atol=1e-5)


def test_ppf() -> None:
    """Test ppf."""
    assert_allclose(mod.ppf(jnp.array([0.25, 0.75])), [0.25, 0.75], atol=1e-5)
