"""Tests for zero_jax.scipy.stats.wrapcauchy."""

from __future__ import annotations

import math

import zero_jax.numpy as jnp
import zero_jax.scipy.stats.wrapcauchy as mod
from tests.conftest import assert_allclose


def test_logpdf() -> None:
    """Test logpdf."""
    assert_allclose(mod.logpdf(jnp.array([0.0]), 0.5), [math.log(3.0 / (2.0 * math.pi))], atol=1e-5)


def test_pdf() -> None:
    """Test pdf."""
    assert_allclose(mod.pdf(jnp.array([0.0]), 0.5), [3.0 / (2.0 * math.pi)], atol=1e-5)
