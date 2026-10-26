"""Tests for zero_jax.scipy.stats.truncnorm."""

from __future__ import annotations

import zero_jax.numpy as jnp
import zero_jax.scipy.stats.truncnorm as mod
from tests.conftest import assert_allclose


def test_cdf() -> None:
    """Test cdf."""
    assert_allclose(mod.cdf(jnp.array([-2.0, 0.0, 2.0]), -1.0, 1.0), [0.0, 0.5, 1.0], atol=1e-4)


def test_logcdf() -> None:
    """Test logcdf."""
    assert_allclose(mod.logcdf(jnp.array([0.0]), -1.0, 1.0), [jnp.log(0.5)], atol=1e-4)


def test_logpdf() -> None:
    """Test logpdf."""
    res = mod.logpdf(jnp.array([0.0]), -1.0, 1.0)
    assert float(res[0].item()) > -0.6


def test_logsf() -> None:
    """Test logsf."""
    assert_allclose(mod.logsf(jnp.array([0.0]), -1.0, 1.0), [jnp.log(0.5)], atol=1e-4)


def test_pdf() -> None:
    """Test pdf."""
    res = mod.pdf(jnp.array([0.0]), -1.0, 1.0)
    assert float(res[0].item()) > 0.5


def test_sf() -> None:
    """Test sf."""
    assert_allclose(mod.sf(jnp.array([0.0]), -1.0, 1.0), [0.5], atol=1e-4)
