"""Tests for zero_jax.scipy.stats."""

from __future__ import annotations

import zero_jax.numpy as jnp
import zero_jax.scipy.stats as mod


def test_gaussian_kde() -> None:
    """Test gaussian_kde."""
    obj = mod.gaussian_kde()
    assert jnp.allclose(obj(jnp.array([1.0, 2.0])), jnp.array([1.0, 2.0]))
    kde = mod.gaussian_kde(jnp.array([0.0, 1.0, 2.0]))
    dens = kde(jnp.array([0.0, 1.0]))
    assert dens.shape == (2,)
    assert float(dens[1].item()) > float(dens[0].item())


def test_mode() -> None:
    """Test mode."""
    m_val, m_cnt = mod.mode(jnp.array([1.0, 2.0, 2.0, 3.0]))
    assert jnp.allclose(m_val, jnp.array(2.0))
    assert jnp.allclose(m_cnt, jnp.array(2))


def test_rankdata() -> None:
    """Test rankdata."""
    ranks = mod.rankdata(jnp.array([30.0, 10.0, 20.0]))
    assert jnp.allclose(ranks, jnp.array([3.0, 1.0, 2.0]))


def test_sem() -> None:
    """Test sem."""
    s = mod.sem(jnp.array([1.0, 2.0, 3.0]))
    assert s.shape == ()
    assert float(s.item()) > 0.5
