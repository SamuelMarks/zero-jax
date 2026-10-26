"""Tests for zero_jax.scipy.cluster.vq."""

from __future__ import annotations

import zero_jax.numpy as jnp
import zero_jax.scipy.cluster.vq as mod
from tests.conftest import assert_allclose


def test_vq() -> None:
    """Test vq."""
    obs = jnp.array([[1.0, 1.0], [5.0, 5.0]])
    code_book = jnp.array([[0.0, 0.0], [6.0, 6.0]])
    codes, dists = mod.vq(obs, code_book)
    assert_allclose(codes, [0, 1])
    assert_allclose(dists, [jnp.sqrt(2.0), jnp.sqrt(2.0)], atol=1e-5)
