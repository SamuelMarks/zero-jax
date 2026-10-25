"""Tests for zero_jax.random.prng."""

from __future__ import annotations

import zero_jax.numpy as jnp
import zero_jax.random.prng as mod


def test_PRNGKey() -> None:
    """Test PRNGKey."""
    k = mod.PRNGKey(42)
    assert k.shape == (2,)


def test_bernoulli() -> None:
    """Test bernoulli."""
    k = mod.PRNGKey(0)
    out = mod.bernoulli(k, 0.5, (3,))
    assert out.shape == (3,)


def test_categorical() -> None:
    """Test categorical."""
    k = mod.PRNGKey(1)
    out = mod.categorical(k, jnp.array([0.2, 0.8]))
    assert out.shape == ()


def test_choice() -> None:
    """Test choice."""
    k = mod.PRNGKey(2)
    out = mod.choice(k, jnp.array([10, 20, 30]), shape=(2,))
    assert out.shape == (2,)


def test_fold_in() -> None:
    """Test fold_in."""
    k = mod.PRNGKey(3)
    out = mod.fold_in(k, 7)
    assert out.shape == (2,)


def test_normal() -> None:
    """Test normal."""
    k = mod.PRNGKey(4)
    out = mod.normal(k, (2, 3))
    assert out.shape == (2, 3)


def test_permutation() -> None:
    """Test permutation."""
    k = mod.PRNGKey(5)
    out = mod.permutation(k, jnp.array([1, 2, 3, 4]))
    assert out.shape == (4,)


def test_randint() -> None:
    """Test randint."""
    k = mod.PRNGKey(6)
    out = mod.randint(k, (2, 2), 0, 10)
    assert out.shape == (2, 2)


def test_split() -> None:
    """Test split."""
    k = mod.PRNGKey(7)
    out = mod.split(k, 3)
    assert out.shape == (3, 2)


def test_truncated_normal() -> None:
    """Test truncated_normal."""
    k = mod.PRNGKey(8)
    out = mod.truncated_normal(k, -2.0, 2.0, (2, 2))
    assert out.shape == (2, 2)


def test_uniform() -> None:
    """Test uniform."""
    k = mod.PRNGKey(9)
    out = mod.uniform(k, (2, 2))
    assert out.shape == (2, 2)
