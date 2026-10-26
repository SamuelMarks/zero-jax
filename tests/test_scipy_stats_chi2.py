"""Tests for zero_jax.scipy.stats.chi2."""

from __future__ import annotations

import zero_jax.numpy as jnp
import zero_jax.scipy.stats.chi2 as mod
from tests.conftest import assert_allclose


def test_cdf() -> None:
    """Test cdf."""
    assert_allclose(mod.cdf(jnp.array([2.0]), 2.0), [1.0 - jnp.exp(-1.0)], atol=1e-4)


def test_logcdf() -> None:
    """Test logcdf."""
    assert_allclose(mod.logcdf(jnp.array([2.0]), 2.0), [jnp.log(1.0 - jnp.exp(-1.0))], atol=1e-4)


def test_logpdf() -> None:
    """Test logpdf."""
    assert_allclose(mod.logpdf(jnp.array([2.0]), 2.0), [-1.0 - jnp.log(2.0)], atol=1e-4)


def test_logsf() -> None:
    """Test logsf."""
    assert_allclose(mod.logsf(jnp.array([2.0]), 2.0), [-1.0], atol=1e-4)


def test_pdf() -> None:
    """Test pdf."""
    assert_allclose(mod.pdf(jnp.array([2.0]), 2.0), [0.5 * jnp.exp(-1.0)], atol=1e-4)


def test_sf() -> None:
    """Test sf."""
    assert_allclose(mod.sf(jnp.array([2.0]), 2.0), [jnp.exp(-1.0)], atol=1e-4)
