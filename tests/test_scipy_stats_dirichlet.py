"""Tests for zero_jax.scipy.stats.dirichlet."""

from __future__ import annotations

import zero_jax.numpy as jnp
import zero_jax.scipy.stats.dirichlet as mod
from tests.conftest import assert_allclose


def test_logpdf() -> None:
    """Test logpdf."""
    assert_allclose(mod.logpdf(jnp.array([0.5, 0.5]), jnp.array([1.0, 1.0])), 0.0, atol=1e-5)


def test_pdf() -> None:
    """Test pdf."""
    assert_allclose(mod.pdf(jnp.array([0.5, 0.5]), jnp.array([1.0, 1.0])), 1.0, atol=1e-5)
