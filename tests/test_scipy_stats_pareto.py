"""Tests for zero_jax.scipy.stats.pareto."""

from __future__ import annotations

import zero_jax.numpy as jnp
import zero_jax.scipy.stats.pareto as mod
from tests.conftest import assert_allclose


def test_logpdf() -> None:
    """Test logpdf."""
    assert_allclose(mod.logpdf(jnp.array([1.0, 2.0]), 2.0), [jnp.log(2.0), jnp.log(0.25)], atol=1e-5)


def test_pdf() -> None:
    """Test pdf."""
    assert_allclose(mod.pdf(jnp.array([1.0, 2.0]), 2.0), [2.0, 0.25], atol=1e-5)
