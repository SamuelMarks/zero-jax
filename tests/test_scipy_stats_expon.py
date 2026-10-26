"""Tests for zero_jax.scipy.stats.expon."""

from __future__ import annotations

import zero_jax.numpy as jnp
import zero_jax.scipy.stats.expon as mod
from tests.conftest import assert_allclose


def test_logpdf() -> None:
    """Test logpdf."""
    assert_allclose(mod.logpdf(jnp.array([0.0, 1.0])), [0.0, -1.0], atol=1e-5)


def test_pdf() -> None:
    """Test pdf."""
    assert_allclose(mod.pdf(jnp.array([0.0, 1.0])), [1.0, jnp.exp(-1.0)], atol=1e-5)
